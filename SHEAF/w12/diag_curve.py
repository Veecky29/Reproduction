"""Diagnostic only (not part of the W1.2 matrix): train one arm and print
train / val / test MSE after every epoch, with early stopping disabled.

Why: every run so far reports best_epoch = 1, i.e. val MSE is minimised after a
single epoch and rises afterwards. If test MSE keeps falling while val rises, the
val split is a misleading selection signal on this dataset and the recipe must
select epochs differently. Prints the whole curve so the decision is visible.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import data as D
import models as M
from train import amp_factory, evaluate, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/autodl-tmp/benchmark")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--arm", default="plain")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--channels", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    set_seed(args.seed)
    win, prior, C = D.build(args.dataset, args.data_root, 96, args.horizon, 1,
                           args.channels, args.seed)
    (xtr, ytr), (xva, yva), (xte, yte) = win["train"], win["val"], win["test"]
    mk = lambda a, b, sh: DataLoader(TensorDataset(torch.from_numpy(a), torch.from_numpy(b)),
                                     batch_size=args.batch, shuffle=sh,
                                     num_workers=4, pin_memory=True)
    ltr, lva, lte = mk(xtr, ytr, True), mk(xva, yva, False), mk(xte, yte, False)

    dev = "cuda"
    model = M.build(args.arm, 96, args.horizon, rho=prior, d=args.d,
                    n_layer=args.layers, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    autocast, scaler = amp_factory(True)

    print(f"{args.dataset} {args.arm} H={args.horizon} lr={args.lr} d={args.d} "
          f"layers={args.layers} params={M.n_params(model):,}", flush=True)
    print(f"{'ep':>3} {'train':>8} {'val':>8} {'test':>8}", flush=True)
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
        v, _ = evaluate(model, lva, dev)
        t, _ = evaluate(model, lte, dev)
        print(f"{ep:3d} {tot / nb:8.4f} {v:8.4f} {t:8.4f}", flush=True)


if __name__ == "__main__":
    main()
