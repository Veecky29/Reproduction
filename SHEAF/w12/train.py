"""W1.2 trainer: one (dataset, arm, horizon, seed) run -> one JSON line.

    python train.py --data-root /root/autodl-tmp/benchmark --dataset ETTh1 \
        --arm sheaf --horizon 96 --seed 2026

Smoke test (128 windows, 2 batches, prints shapes):
    python train.py --smoke --dataset ETTh1 --arm sheaf --horizon 96
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import data as D
import models as M


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def amp_factory(use_amp: bool):
    """New (torch>=2.3) and legacy AMP entry points, whichever exists."""
    if hasattr(torch.amp, "GradScaler"):
        return (lambda: torch.amp.autocast("cuda", enabled=use_amp),
                torch.amp.GradScaler("cuda", enabled=use_amp))
    return (lambda: torch.cuda.amp.autocast(enabled=use_amp),
            torch.cuda.amp.GradScaler(enabled=use_amp))


def evaluate(model, loader, device):
    model.eval()
    sq = abs_ = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            y_norm, mu, sd = model(xb)
            pred = (y_norm * sd + mu).transpose(1, 2)      # (B,C,H)
            tgt = yb.transpose(1, 2)                       # (B,C,H)
            d = pred - tgt
            sq += float((d ** 2).sum())
            abs_ += float(d.abs().sum())
            n += d.numel()
    return sq / n, abs_ / n


def run(args):
    set_seed(args.seed)
    dev = args.device

    win, prior, C = D.build(args.dataset, args.data_root, args.L, args.horizon,
                           args.stride, args.channels, args.seed,
                           drop=[c for c in args.drop_channels.split(",") if c.strip()])
    (xtr, ytr), (xva, yva), (xte, yte) = win["train"], win["val"], win["test"]
    if args.smoke:
        xtr, ytr = xtr[:128], ytr[:128]
        xva, yva = xva[:64], yva[:64]
        xte, yte = xte[:64], yte[:64]
    print(f"    windows train/val/test = {len(xtr)}/{len(xva)}/{len(xte)}  "
          f"C={C}  L={args.L} H={args.horizon}", flush=True)

    mk = lambda a, b, sh: DataLoader(
        TensorDataset(torch.from_numpy(a), torch.from_numpy(b)),
        batch_size=args.batch, shuffle=sh,
        num_workers=args.workers, pin_memory=(dev == "cuda"))

    ltr, lva, lte = mk(xtr, ytr, True), mk(xva, yva, False), mk(xte, yte, False)

    model = M.build(args.arm, args.L, args.horizon, rho=prior, d=args.d,
                    n_layer=args.layers, patch=args.patch,
                    dropout=args.dropout, C=int(C), beta_init=args.beta_init).to(dev)
    print(f"    arm={args.arm}  params={M.n_params(model):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    use_amp = (dev == "cuda")
    autocast, scaler = amp_factory(use_amp)

    best = dict(val=float("inf"), epoch=-1, state=None)
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in ltr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                y_norm, mu, sd = model(xb)
                loss = nn.functional.mse_loss(y_norm, (yb - mu) / sd)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
            if args.smoke and nb >= 2:
                break
        sched.step()
        vmse, _ = evaluate(model, lva, dev)
        hist.append(dict(epoch=ep, train=round(tot / max(nb, 1), 6), val=round(vmse, 6)))
        print(f"    ep{ep:02d} train={tot / max(nb, 1):.5f} val={vmse:.5f}", flush=True)
        if vmse < best["val"] - 1e-5:
            best = dict(val=vmse, epoch=ep,
                        state=copy.deepcopy({k: v.detach().cpu()
                                             for k, v in model.state_dict().items()}))
        if not args.fixed_epochs and ep - best["epoch"] >= args.patience:
            print(f"    early stop at ep{ep}", flush=True)
            break

    # fixed-epochs mode: keep the LAST epoch, not the best-val one. Val on these
    # benchmarks bottoms out in epoch 1-3, so best-val selection adds ~15% seed
    # variance and swamps every arm comparison.
    if not args.fixed_epochs and best["state"] is not None:
        model.load_state_dict(best["state"])
    tmse, tmae = evaluate(model, lte, dev)

    rec = dict(dataset=args.dataset, arm=args.arm, horizon=args.horizon,
               seed=args.seed, L=args.L, C=int(C), params=M.n_params(model),
               best_epoch=best["epoch"], val_mse=round(best["val"] if not args.fixed_epochs else hist[-1]["val"], 6),
               test_mse=round(tmse, 6), test_mae=round(tmae, 6),
               seconds=round(time.time() - t0, 1), d=args.d, layers=args.layers,
               patch=args.patch, batch=args.batch, lr=args.lr, epochs_run=len(hist),
               beta_init=args.beta_init)
    # read-out diagnostics: whether the cross-channel branch actually opened is the
    # first thing to look at before blaming (or crediting) the architecture.
    if hasattr(model, "msg_scale"):
        rec.update(alpha=round(float(model.msg_scale.detach()), 5),
                   delta=round(float(model.delta.detach()), 5),
                   eta=round(float(model.eta.detach()), 5),
                   tau=round(float(torch.exp(model.log_tau.detach())), 5))
    if hasattr(model, "beta"):
        b = model.beta.detach()
        rec.update(beta_mean=round(float(b.mean()), 5),
                   beta_min=round(float(b.min()), 5),
                   beta_max=round(float(b.max()), 5))
        gs = getattr(model, "gate_stats", None)
        if gs:
            rec.update(gate_mean=round(gs[0], 5), gate_sd=round(gs[1], 5),
                       gate_max=round(gs[2], 5))
    print("  " + json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/autodl-tmp/benchmark")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--arm", default="sheaf",
                    choices=["plain", "nostats", "film", "concat", "mix", "sheaf"])
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--channels", type=int, default=40,
                    help="subsample channels above this; Electricity -> 40 (checkB parity)")
    ap.add_argument("--drop-channels", default="",
                    help="comma-separated column names to remove, e.g. 'rain (mm),raining (s)'")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--beta-init", type=float, default=0.1,
                    help="initial value of the per-channel mixing gate (mix arm)")
    ap.add_argument("--fixed-epochs", action="store_true",
                    help="no early stopping; keep the last epoch (low-variance protocol)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/w12.jsonl")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"[{args.dataset} {args.arm} H={args.horizon} seed={args.seed} "
          f"dev={args.device}]", flush=True)
    rec = run(args)
    if not args.smoke:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()




