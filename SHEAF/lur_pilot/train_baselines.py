#!/usr/bin/env python3
"""Train DLinear and iTransformer with the official TS_Predictability entry points.

This does not start by itself in CI. One dataset-model pair is a full GPU run
(default 10 epochs, the upstream early-stopping patience).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import phase0


def _run(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        text = log_path.read_text(errors="replace")
        saved = "Saving model" in text
        # DLinear/exp/exp_main.py reloads the saved checkpoint with
        # map_location='cuda:0' after early stopping. The file is already
        # written. On a machine without CUDA that reload is the whole failure.
        cuda_reload = saved and "Attempting to deserialize object on a CUDA device" in text
        # iTransformer/run.py calls exp.test() after training. test() stacks
        # batches with np.array, which fails when the last batch is shorter.
        # The checkpoint is written before that call. evaluate_predictability
        # does not use this stacking.
        uneven_test = saved and "inhomogeneous shape" in text
        if cuda_reload or uneven_test:
            print(
                "Official training saved the checkpoint, then exited in a step "
                "Phase 0 does not use: DLinear reloads weights onto cuda:0, and "
                "iTransformer test() cannot stack a shorter final batch. "
                "Diagnosis loads the saved checkpoint directly."
            )
            return
        lines = text.splitlines()
        tail = "\n".join(lines[-50:])
        raise SystemExit(
            f"Training failed (exit {proc.returncode}).\n"
            f"Log: {log_path}\n--- tail ---\n{tail}"
        )


def main() -> None:
    args = phase0.build_parser().parse_args()
    cfg = phase0.load_config(Path(args.config))
    phase0.require_upstream()
    manifest = phase0.read_manifest()
    datasets = phase0.selected_datasets(cfg, args)
    models = phase0.selected_models(args)
    for dataset in datasets:
        if dataset not in manifest["datasets"]:
            raise SystemExit(
                f"{dataset} is not in the manifest. Run: python prepare_data.py "
                f"--dataset {dataset}"
            )
    for dataset in datasets:
        for model in models:
            spec = phase0.make_spec(dataset, model, manifest, cfg, args)
            if int(spec.use_ps_loss) != 0:
                raise SystemExit("Phase 0 trains with MSE only (use_ps_loss must be 0).")
            script = phase0.model_dir(model) / phase0.script_for(model)
            if not script.is_file():
                raise SystemExit(f"Official entry point missing: {script}")
            cmd = [sys.executable, "-u", str(script), *spec.argv(is_training=1)]
            log_path = phase0.work_dir() / "logs" / f"train_{dataset}_{model}_H{spec.horizon}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
            env.setdefault("MPLBACKEND", "Agg")
            print(f"[{dataset} / {model}] setting {spec.setting()}")
            print(f"[{dataset} / {model}] log {log_path}")
            print(" ".join(cmd))
            if args.dry_run:
                continue
            Path(spec.checkpoints).mkdir(parents=True, exist_ok=True)
            _run(cmd, cwd=phase0.model_dir(model), env=env, log_path=log_path)
            ckpt = Path(spec.checkpoints) / spec.setting() / "checkpoint.pth"
            if not ckpt.is_file():
                raise SystemExit(
                    f"Training returned but checkpoint is missing: {ckpt}\n"
                    f"The setting string did not match the official trainer."
                )
            print(f"[{dataset} / {model}] checkpoint {ckpt}")
    if args.dry_run:
        print("dry-run: no training started")


if __name__ == "__main__":
    main()
