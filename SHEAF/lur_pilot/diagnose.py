#!/usr/bin/env python3
"""Run official SCP / MSE_lb / band LUR and write the Phase 0 tables.

Forecasts come from checkpoints produced by train_baselines.py. The numbers
come from evaluate_predictability and evaluate_band_LUR in the cloned
TS_Predictability tree. This file only averages the official columns and
applies the pre-registered stop rule.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import phase0


def _read_overall(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected one row in {path}, found {len(rows)}.")
    return rows[0]


def _read_band_summary(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty band summary: {path}")
    return rows


def _float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Column {key!r} missing or not numeric in official CSV.") from exc


def _run_one(spec: phase0.RunSpec, args, cfg: dict) -> dict:
    record = {
        "dataset": spec.dataset,
        "model": spec.model,
        "horizon": spec.horizon,
        "setting": spec.setting(),
        "enc_in": spec.enc_in,
    }
    channels = phase0.parse_channel_spec(str(cfg["lur_channels"]), spec.enc_in)
    lur_root = phase0.work_dir() / "official" / spec.dataset / spec.model / f"H{spec.horizon}"
    scp_dir = lur_root / "scp"
    lur_dir = lur_root / "lur"
    cmd = [
        sys.executable,
        "-u",
        str(phase0.ROOT / "diagnose_one.py"),
        "--model-dir",
        str(phase0.model_dir(spec.model)),
        "--script",
        phase0.script_for(spec.model),
        "--setting",
        spec.setting(),
        "--task",
        "both",
        "--channels",
        ",".join(str(c) for c in channels),
        "--lur-bands",
        str(cfg["lur_bands"]),
        "--lur-max-samples",
        str(cfg["lur_max_samples"]),
        "--welch-win-frac",
        str(cfg["welch_win_frac"]),
        "--welch-overlap",
        str(cfg["welch_overlap"]),
        "--alpha-boundary",
        str(cfg["alpha_boundary"]),
        "--outdir-scp",
        str(scp_dir),
        "--outdir-lur",
        str(lur_dir),
    ]
    if cfg.get("lur_include_dc"):
        cmd.append("--include-dc")
    if args.limit_batches is not None:
        cmd.extend(["--limit-batches", str(args.limit_batches)])
    cmd.append("--")
    cmd.extend(spec.argv(is_training=0))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
    env["MPLBACKEND"] = "Agg"
    # The child inserts the official model directory on sys.path itself.
    # Dropping this package's directory from PYTHONPATH avoids shadowing
    # official modules if a parent exported it.
    env.pop("PYTHONPATH", None)
    print(f"[{spec.dataset} / {spec.model}] diagnose setting {spec.setting()}")
    print(f"[{spec.dataset} / {spec.model}] LUR channels {channels}")
    if args.dry_run:
        print(" ".join(cmd))
        record["dry_run"] = True
        record["channels"] = channels
        return record
    proc = subprocess.Popen(
        cmd,
        cwd=phase0.model_dir(spec.model),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    payload = None
    tail: list[str] = []
    for line in proc.stdout:
        tail.append(line)
        if len(tail) > 80:
            tail.pop(0)
        if line.startswith("PHASE0_JSON "):
            payload = json.loads(line[len("PHASE0_JSON ") :])
        else:
            print(line, end="")
    code = proc.wait()
    if code != 0:
        raise SystemExit(
            f"Diagnosis failed for {spec.dataset} / {spec.model} (exit {code}).\n"
            f"--- tail ---\n{''.join(tail)}"
        )
    if payload is None:
        raise SystemExit(
            f"diagnose_one.py exited 0 without PHASE0_JSON for {spec.dataset} / {spec.model}.\n"
            f"--- tail ---\n{''.join(tail)}"
        )
    overall = _read_overall(Path(payload["scp_overall_csv"]))
    per_channel = []
    for item in payload["lur_channels"]:
        rows = _read_band_summary(Path(item["summary_csv"]))
        per_channel.append({"channel": item["channel"], "rows": rows, "summary_csv": item["summary_csv"]})
    bands = phase0.aggregate_bands(per_channel)
    channel_overall = []
    for item in per_channel:
        channel_overall.append(
            {
                "channel": item["channel"],
                "overall_lur": phase0.weighted_lur(
                    [float(row["XR_mean"]) for row in item["rows"]],
                    [float(row["weight_mean"]) for row in item["rows"]],
                ),
            }
        )
    overall_lur = phase0.nanmean([row["overall_lur"] for row in channel_overall])
    record.update(
        {
            "channels": [item["channel"] for item in per_channel],
            "scp": _float(overall, "P_lin_mean_overall"),
            "mse_lb": _float(overall, "mse_lb_mean_overall"),
            "nmse_lb": _float(overall, "nmse_lb_mean_overall"),
            "mse_model": _float(overall, "mse_model_mean_overall"),
            "mae_model": _float(overall, "mae_model_mean_overall"),
            "p_lin": _float(overall, "P_lin_mean_overall"),
            "overall_lur": overall_lur,
            "bands": bands,
            "channel_overall_lur": channel_overall,
            "official_scp_csv": payload["scp_overall_csv"],
            "official_lur": payload["lur_channels"],
        }
    )
    return record


def _effective_protocol(cfg: dict, args, manifest: dict, datasets: list[str]) -> dict:
    kill = cfg.get("kill_rule") or {}
    horizon = args.horizon if args.horizon is not None else int(manifest.get("horizon", cfg["horizon"]))
    return {
        "horizon": horizon,
        "datasets": tuple(datasets),
        "kill_model": str(kill.get("model", "iTransformer")),
        "kill_datasets": tuple(kill.get("datasets") or phase0.PREREGISTERED["kill_datasets"]),
        "lur_bands": args.lur_bands if args.lur_bands is not None else cfg.get("lur_bands"),
        "lur_max_samples": args.lur_max_samples if args.lur_max_samples is not None else int(cfg.get("lur_max_samples")),
        "lur_channels": args.lur_channels if args.lur_channels is not None else cfg.get("lur_channels"),
        "lur_include_dc": bool(cfg.get("lur_include_dc", False)),
        "welch_win_frac": float(cfg.get("welch_win_frac")),
        "welch_overlap": float(cfg.get("welch_overlap")),
        "alpha_boundary": float(cfg.get("alpha_boundary")),
        "xr_below": float(kill.get("xr_below")),
        "min_energy_weight": float(kill.get("min_energy_weight")),
        "weather_drop_rain": bool(manifest.get("weather_drop_rain")),
    }


def _apply_cli_overrides(cfg: dict, args) -> dict:
    merged = dict(cfg)
    if args.lur_channels is not None:
        merged["lur_channels"] = args.lur_channels
    if args.lur_bands is not None:
        merged["lur_bands"] = args.lur_bands
    if args.lur_max_samples is not None:
        merged["lur_max_samples"] = args.lur_max_samples
    return merged


def _write_tables(rows: list[dict], protocol: dict, verdict: dict) -> None:
    out = phase0.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "phase0_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        fields = [
            "dataset",
            "model",
            "horizon",
            "enc_in",
            "SCP",
            "MSE_lb",
            "NMSE_lb",
            "MSE_model",
            "MAE_model",
            "LUR_overall",
            "n_lur_channels",
            "setting",
            "official_scp_csv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("dry_run"):
                continue
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "horizon": row["horizon"],
                    "enc_in": row["enc_in"],
                    "SCP": row["scp"],
                    "MSE_lb": row["mse_lb"],
                    "NMSE_lb": row["nmse_lb"],
                    "MSE_model": row["mse_model"],
                    "MAE_model": row["mae_model"],
                    "LUR_overall": row["overall_lur"],
                    "n_lur_channels": len(row["channels"]),
                    "setting": row["setting"],
                    "official_scp_csv": row["official_scp_csv"],
                }
            )
    bands_path = out / "phase0_bands.csv"
    with bands_path.open("w", newline="") as handle:
        fields = ["dataset", "model", "channel", "band", "LUR", "energy_weight"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("dry_run"):
                continue
            for band in row["bands"]:
                writer.writerow(
                    {
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "channel": "mean",
                        "band": band["band"],
                        "LUR": band["lur_mean"],
                        "energy_weight": band["weight_mean"],
                    }
                )
            for item in row["official_lur"]:
                for band_row in _read_band_summary(Path(item["summary_csv"])):
                    writer.writerow(
                        {
                            "dataset": row["dataset"],
                            "model": row["model"],
                            "channel": item["channel"],
                            "band": band_row["band"],
                            "LUR": band_row["XR_mean"],
                            "energy_weight": band_row["weight_mean"],
                        }
                    )
    summary = {
        "protocol": protocol,
        "preregistered_protocol": verdict["preregistered_protocol"],
        "verdict": verdict["verdict"],
        "text": verdict["text"],
        "kill_rule": {
            "model": protocol["kill_model"],
            "datasets": list(protocol["kill_datasets"]),
            "xr_below": protocol["xr_below"],
            "min_energy_weight": protocol["min_energy_weight"],
            "definition": (
                "SCP is official P_lin_mean_overall. MSE_lb is official "
                "mse_lb_mean_overall. Band LUR is the mean of official XR_mean "
                "across the listed channels. Overall LUR is the mean, across "
                "channels, of the energy-weighted average of official XR_mean."
            ),
        },
        "datasets": verdict["datasets"],
        "runs": [
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "SCP": row.get("scp"),
                "MSE_lb": row.get("mse_lb"),
                "LUR_overall": row.get("overall_lur"),
                "channels": row.get("channels"),
                "setting": row.get("setting"),
            }
            for row in rows
            if not row.get("dry_run")
        ],
    }
    json_path = out / "phase0_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path = out / "phase0_summary.md"
    md_path.write_text(_markdown(summary, rows))
    print(f"wrote {metrics_path}")
    print(f"wrote {bands_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"verdict: {verdict['verdict']}")
    print(verdict["text"])


def _markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Phase 0 summary",
        "",
        summary["text"],
        "",
        f"Verdict: `{summary['verdict']}`. Pre-registered protocol: `{summary['preregistered_protocol']}`.",
        "",
        "SCP is the official `P_lin_mean_overall`. MSE_lb is the official `mse_lb_mean_overall`. LUR_overall averages the official band `XR_mean` with the official energy weights.",
        "",
        "| Dataset | Model | Horizon | SCP | MSE_lb | MSE_model | LUR_overall |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("dry_run"):
            continue
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['horizon']} | "
            f"{row['scp']:.6f} | {row['mse_lb']:.6f} | {row['mse_model']:.6f} | {row['overall_lur']:.6f} |"
        )
    lines.extend(["", "## Bands (mean over evaluated channels)", ""])
    lines.append("| Dataset | Model | Band | LUR | Energy weight |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        if row.get("dry_run"):
            continue
        for band in row["bands"]:
            lines.append(
                f"| {row['dataset']} | {row['model']} | {band['band']} | "
                f"{band['lur_mean']:.6f} | {band['weight_mean']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = phase0.build_parser().parse_args()
    cfg = phase0.load_config(Path(args.config))
    cfg = _apply_cli_overrides(cfg, args)
    phase0.require_upstream()
    manifest = phase0.read_manifest()
    datasets = phase0.selected_datasets(cfg, args)
    models = phase0.selected_models(args)
    for dataset in datasets:
        if dataset not in manifest["datasets"]:
            raise SystemExit(
                f"{dataset} is not in the manifest. Run python prepare_data.py --dataset {dataset}"
            )
    rows = []
    for dataset in datasets:
        for model in models:
            spec = phase0.make_spec(dataset, model, manifest, cfg, args)
            rows.append(_run_one(spec, args, cfg))
    if args.dry_run:
        print("dry-run: no diagnosis started")
        return
    protocol = _effective_protocol(cfg, args, manifest, datasets)
    kill_model = protocol["kill_model"]
    by_dataset = {}
    for row in rows:
        if row["model"] == kill_model:
            by_dataset[row["dataset"]] = row["bands"]
    verdict = phase0.kill_verdict(
        by_dataset,
        protocol_ok=phase0.protocol_is_preregistered(protocol),
        xr_below=float(protocol["xr_below"]),
        min_weight=float(protocol["min_energy_weight"]),
    )
    _write_tables(rows, protocol, verdict)


if __name__ == "__main__":
    main()
