#!/usr/bin/env python3
"""Call official evaluate_predictability and evaluate_band_LUR for one run.

Launch this with PYTHONPATH left alone. It executes the official trainer
script only far enough to reuse that script's argument parser, then calls
the two diagnostic methods. Do not import this from a long-lived process
that has already imported a different model's `exp` or `utils` package.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def _parse_our_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="One official SCP/LUR call")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--task", choices=("scp", "lur", "both"), default="both")
    parser.add_argument("--channels", required=True, help="Comma-separated channel ids.")
    parser.add_argument("--lur-bands", required=True)
    parser.add_argument("--lur-max-samples", type=int, required=True)
    parser.add_argument("--include-dc", action="store_true")
    parser.add_argument("--welch-win-frac", type=float, required=True)
    parser.add_argument("--welch-overlap", type=float, required=True)
    parser.add_argument("--alpha-boundary", type=float, required=True)
    parser.add_argument("--outdir-scp", required=True)
    parser.add_argument("--outdir-lur", required=True)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("official", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    official = list(args.official)
    if official and official[0] == "--":
        official = official[1:]
    if not official:
        parser.error("missing official argv after --")
    return args, official


def _parse_official(model_dir: Path, script: str, official_argv: list[str]):
    import argparse as argp

    captured: dict = {}
    real = argp.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        if args is None:
            args = official_argv
        namespace = real(self, args=args, namespace=namespace)
        captured["ns"] = namespace
        raise SystemExit(0)

    argp.ArgumentParser.parse_args = grab
    old_argv = sys.argv
    sys.argv = [str(model_dir / script), *official_argv]
    try:
        runpy.run_path(str(model_dir / script), run_name="__main__")
    except SystemExit as exc:
        if "ns" not in captured:
            raise
        if exc.code not in (0, None):
            raise
    finally:
        argp.ArgumentParser.parse_args = real
        sys.argv = old_argv
    return captured["ns"]


def _exp_class(model_name: str):
    if model_name == "iTransformer":
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

        return Exp_Long_Term_Forecast
    if model_name == "DLinear":
        from exp.exp_main import Exp_Main

        return Exp_Main
    raise SystemExit(f"No official experiment class for {model_name}.")


def _newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args, official_argv = _parse_our_args(sys.argv[1:])
    model_dir = Path(args.model_dir).resolve()
    os.chdir(model_dir)
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    ns = _parse_official(model_dir, args.script, official_argv)
    import torch

    ns.use_gpu = bool(torch.cuda.is_available() and ns.use_gpu)
    if getattr(ns, "use_multi_gpu", False) and ns.use_gpu:
        device_ids = [int(part) for part in str(ns.devices).split(",") if part.strip() != ""]
        ns.device_ids = device_ids
        if device_ids:
            ns.gpu = device_ids[0]

    ckpt = Path(ns.checkpoints) / args.setting / "checkpoint.pth"
    if not ckpt.is_file():
        raise SystemExit(
            f"Checkpoint not found: {ckpt}\n"
            f"Train this setting with train_baselines.py before diagnose.py."
        )

    Exp = _exp_class(ns.model)
    exp = Exp(ns)
    channels = [int(part) for part in args.channels.split(",") if part.strip() != ""]
    report: dict = {
        "setting": args.setting,
        "checkpoint": str(ckpt),
        "model": ns.model,
        "data": ns.data,
        "data_path": ns.data_path,
    }

    if args.task in ("scp", "both"):
        outdir = Path(args.outdir_scp)
        outdir.mkdir(parents=True, exist_ok=True)
        exp.evaluate_predictability(
            setting=args.setting,
            load=True,
            alpha_boundary=args.alpha_boundary,
            welch_win_frac=args.welch_win_frac,
            welch_overlap=args.welch_overlap,
            workers=min(8, os.cpu_count() or 1),
            outdir=str(outdir),
            limit_batches=args.limit_batches,
        )
        matches = list(outdir.rglob("dataset_overall_*.csv"))
        if not matches:
            raise SystemExit(f"evaluate_predictability wrote no dataset_overall CSV under {outdir}")
        overall = _newest(matches)
        report["scp_overall_csv"] = str(overall)
        report["scp_dir"] = str(overall.parent)

    if args.task in ("lur", "both"):
        outdir = Path(args.outdir_lur)
        outdir.mkdir(parents=True, exist_ok=True)
        band_paths = []
        for channel in channels:
            paths = exp.evaluate_band_LUR(
                setting=args.setting,
                channel=channel,
                bands=args.lur_bands,
                welch_win_frac=args.welch_win_frac,
                welch_overlap=args.welch_overlap,
                include_dc=args.include_dc,
                alpha_boundary=args.alpha_boundary,
                max_samples=args.lur_max_samples,
                outdir=str(outdir),
                load=True,
                ckpt_file=str(ckpt),
            )
            if not paths or "summary_csv" not in paths:
                raise SystemExit(
                    f"evaluate_band_LUR returned no summary for channel {channel}. "
                    f"Got: {paths!r}"
                )
            band_paths.append({"channel": channel, "summary_csv": paths["summary_csv"], "detail_csv": paths.get("detail_csv")})
        report["lur_channels"] = band_paths

    print("PHASE0_JSON " + json.dumps(report))


if __name__ == "__main__":
    main()
