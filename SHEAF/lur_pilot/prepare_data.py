#!/usr/bin/env python3
"""Resolve TSLib CSVs and write SHEAF/lur_pilot/work/data.

Weather rain / raining columns are dropped here, before any model sees them.
Traffic is not skipped: a missing file exits with the paths that were searched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import phase0


def main() -> None:
    args = phase0.build_parser().parse_args()
    cfg = phase0.load_config(Path(args.config))
    root = phase0.data_root_from(cfg, args.data_root)
    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")
    datasets = phase0.selected_datasets(cfg, args)
    horizon = args.horizon if args.horizon is not None else int(cfg["horizon"])
    target = str(cfg.get("target") or "OT")
    drop_rain = bool(cfg.get("weather_drop_rain", True))
    dest_dir = phase0.work_dir() / "data"
    print(f"data_root: {root}")
    print(f"datasets: {', '.join(datasets)}")
    print(f"horizon: {horizon}")
    print(f"weather_drop_rain: {drop_rain}")
    records = {}
    for dataset in datasets:
        source = phase0.resolve_csv(root, dataset)
        print(f"[{dataset}] source {source}")
        if args.dry_run:
            header = phase0.read_header(source)
            rain = [name for name in header if phase0.is_rain_column(name)]
            print(
                f"[{dataset}] value columns {phase0.encoder_count(header)}; "
                f"rain columns {rain or 'none'}"
            )
            continue
        records[dataset] = phase0.prepare_one(
            dataset,
            source,
            dest_dir,
            target=target,
            drop_rain=drop_rain,
            allow_channel_mismatch=args.allow_channel_mismatch,
        )
        rec = records[dataset]
        print(
            f"[{dataset}] prepared {rec['prepared']} enc_in={rec['enc_in']} "
            f"dropped={rec['columns_dropped'] or 'none'}"
        )
    if args.dry_run:
        print("dry-run: manifest not written")
        return
    payload = {
        "data_root": str(root),
        "horizon": horizon,
        "target": target,
        "weather_drop_rain": drop_rain,
        "datasets": records,
        "note": (
            "Weather column drops are a data-quality control, not a forecasting method. "
            "ETT is absent unless it was requested."
        ),
    }
    phase0.write_manifest(payload)
    print(f"wrote {phase0.manifest_path()}")
    print(
        json.dumps(
            {
                key: {"enc_in": value["enc_in"], "dropped": value["columns_dropped"]}
                for key, value in records.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
