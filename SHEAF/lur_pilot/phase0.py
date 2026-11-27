"""Shared config, dataset prep, official-script argv, and kill rule.

SCP / MSE_lb / LUR themselves are computed by the cloned TS_Predictability
code. This module only prepares inputs and reads the official CSV columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "3bd624602c95506b6f9beedfc43c041fdb6afb36"
UPSTREAM_URL = "https://github.com/WanjinVon/TS_Predictability.git"

DEFAULT_DATASETS = ("electricity", "traffic", "weather")
ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
MODELS = ("DLinear", "iTransformer")

# Fixed before any Phase 0 run of this package. diagnose.py compares the
# effective settings with this dict and will not emit a pre-registered
# "stop" verdict when they differ.
PREREGISTERED = {
    "horizon": 96,
    "datasets": DEFAULT_DATASETS,
    "kill_model": "iTransformer",
    "kill_datasets": ("electricity", "traffic"),
    "lur_bands": "auto:10",
    "lur_max_samples": 1000,
    "lur_channels": "even:16",
    "lur_include_dc": False,
    "welch_win_frac": 0.25,
    "welch_overlap": 0.5,
    "alpha_boundary": 1.0,
    "xr_below": 0.5,
    "min_energy_weight": 0.05,
    "weather_drop_rain": True,
}

EXPECTED_ENCODERS = {
    "electricity": 321,
    "traffic": 862,
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
}

DATA_FILE = {
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
}

# Official loader key (--data). ECL/weather/traffic use the custom reader.
DATA_KEY = {
    "electricity": "custom",
    "traffic": "custom",
    "weather": "custom",
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
}

# model_id prefixes copied from the upstream shells. Traffic is not in those
# shells; both models use the TSLib script prefix "traffic".
MODEL_ID_PREFIX = {
    ("electricity", "iTransformer"): "ECL",
    ("electricity", "DLinear"): "Electricity",
    ("traffic", "iTransformer"): "traffic",
    ("traffic", "DLinear"): "traffic",
    ("weather", "iTransformer"): "weather",
    ("weather", "DLinear"): "weather",
    ("ETTh1", "iTransformer"): "ETTh1",
    ("ETTh1", "DLinear"): "ETTh1",
    ("ETTh2", "iTransformer"): "ETTh2",
    ("ETTh2", "DLinear"): "ETTh2",
    ("ETTm1", "iTransformer"): "ETTm1",
    ("ETTm1", "DLinear"): "ETTm1",
    ("ETTm2", "iTransformer"): "ETTm2",
    ("ETTm2", "DLinear"): "ETTm2",
}

_RAIN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def work_dir() -> Path:
    return ROOT / "work"


def results_dir() -> Path:
    return ROOT / "results"


def upstream_dir() -> Path:
    return ROOT / "third_party" / "TS_Predictability"


def manifest_path() -> Path:
    return work_dir() / "data" / "manifest.json"


def is_rain_column(name: str) -> bool:
    """True for rain / raining, including 'rain (mm)' and 'raining (s)'."""
    raw = name.strip().lower()
    base = _RAIN_RE.sub("", raw).strip()
    base = re.sub(r"\s+", " ", base)
    return base in {"rain", "raining"}


def even_channels(n: int, k: int) -> list[int]:
    if n <= 0:
        raise ValueError(f"channel count must be positive, got {n}")
    if k <= 0:
        raise ValueError(f"channel sample size must be positive, got {k}")
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    chosen: list[int] = []
    for i in range(k):
        idx = int(round(i * (n - 1) / (k - 1)))
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def parse_channel_spec(spec: str, n: int) -> list[int]:
    text = spec.strip()
    if text == "all":
        return list(range(n))
    if text.startswith("even:"):
        return even_channels(n, int(text.split(":", 1)[1]))
    channels = [int(part) for part in text.split(",") if part.strip() != ""]
    if not channels:
        raise ValueError(f"empty channel spec: {spec!r}")
    bad = [c for c in channels if c < 0 or c >= n]
    if bad:
        raise ValueError(f"channels {bad} outside 0..{n - 1}")
    return channels


def normalize_dataset_name(name: str) -> str:
    key = name.strip()
    aliases = {
        "ecl": "electricity",
        "electricity": "electricity",
        "traffic": "traffic",
        "weather": "weather",
        "etth1": "ETTh1",
        "etth2": "ETTh2",
        "ettm1": "ETTm1",
        "ettm2": "ETTm2",
    }
    found = aliases.get(key.lower())
    if found is None:
        known = ", ".join((*DEFAULT_DATASETS, *ETT_DATASETS))
        raise SystemExit(f"Unknown dataset {name!r}. Known names: {known}.")
    return found


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "config.yaml")
    with cfg_path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config is not a mapping: {cfg_path}")
    return cfg


def data_root_from(cfg: dict, cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env = os.environ.get("LUR_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    raw = str(cfg.get("data_root") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    raise SystemExit(
        "No dataset directory. Set LUR_DATA_ROOT, or data_root in "
        "SHEAF/lur_pilot/config.yaml, or pass --data-root."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 SCP / MSE_lb / LUR")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated names.")
    parser.add_argument("--model", action="append", default=None, choices=list(MODELS))
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--include-ett", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-channel-mismatch", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--lur-channels", default=None)
    parser.add_argument("--lur-bands", default=None)
    parser.add_argument("--lur-max-samples", type=int, default=None)
    return parser


def selected_datasets(cfg: dict, args: argparse.Namespace) -> list[str]:
    if args.dataset and args.datasets:
        raise SystemExit("Pass either --dataset or --datasets, not both.")
    explicit = bool(args.dataset or args.datasets)
    if args.datasets:
        names = [normalize_dataset_name(part) for part in args.datasets.split(",")]
    elif args.dataset:
        names = [normalize_dataset_name(part) for part in args.dataset]
    else:
        names = [normalize_dataset_name(part) for part in cfg.get("datasets", DEFAULT_DATASETS)]
    add_ett = args.include_ett or (not explicit and bool(cfg.get("include_ett")))
    if add_ett:
        for name in ETT_DATASETS:
            if name not in names:
                names.append(name)
    if not names:
        raise SystemExit("No datasets selected.")
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    ett = [name for name in seen if name in ETT_DATASETS]
    # ETT stays out of the default run. Naming it on the CLI, or include_ett, opts in.
    if ett and not (args.include_ett or cfg.get("include_ett") or explicit):
        raise SystemExit(
            "ETT is not in the default Phase 0 run. Pass --include-ett, or name "
            "the ETT set explicitly with --dataset."
        )
    return seen


def selected_models(args: argparse.Namespace) -> list[str]:
    if not args.model:
        return list(MODELS)
    seen: list[str] = []
    for name in args.model:
        if name not in seen:
            seen.append(name)
    return seen


def search_candidates(dataset: str) -> list[str]:
    filename = DATA_FILE[dataset]
    stem = Path(filename).stem
    lower = stem.lower()
    paths = [
        f"{lower}/{filename}",
        f"{lower}/{lower}.csv",
        filename,
        f"{stem}.csv",
        f"{lower}.csv",
        f"{stem}/{filename}",
    ]
    if dataset == "electricity":
        paths.extend(
            [
                "electricity/electricity.csv",
                "Electricity/Electricity.csv",
                "Electricity.csv",
                "ECL/electricity.csv",
                "ECL/ECL.csv",
            ]
        )
    if dataset in ETT_DATASETS:
        paths.extend(
            [
                f"ETT-small/{filename}",
                f"ETT/{filename}",
                f"ett/{filename}",
            ]
        )
    # Stable unique order.
    unique: list[str] = []
    for item in paths:
        if item not in unique:
            unique.append(item)
    return unique


def resolve_csv(data_root: Path, dataset: str) -> Path:
    tried = []
    for relative in search_candidates(dataset):
        path = data_root / relative
        tried.append(str(path))
        if path.is_file():
            return path
    lines = "\n".join(f"  - {item}" for item in tried)
    raise SystemExit(
        f"{dataset} CSV not found under {data_root}.\n"
        f"Phase 0 does not skip a selected dataset.\n"
        f"Searched:\n{lines}\n"
        f"Use TSLib names (electricity.csv, traffic.csv, weather.csv) "
        f"or the capitalized names in SHEAF/data (Electricity.csv, Traffic.csv, Weather.csv). "
        f"Set LUR_DATA_ROOT or data_root in config.yaml."
    )


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SystemExit(f"Empty CSV: {path}") from exc
    if not header:
        raise SystemExit(f"CSV has an empty header: {path}")
    return header


def require_columns(header: list[str], path: Path, target: str) -> None:
    if "date" not in header:
        raise SystemExit(
            f"{path} has no 'date' column. The official loader expects a TSLib CSV "
            f"whose first columns include date. Header starts: {header[:8]}"
        )
    if target not in header:
        raise SystemExit(
            f"{path} has no target column {target!r}. The official Dataset_Custom "
            f"loader requires it (features=M still uses every series; the target "
            f"column is only moved to the end). Pass a CSV that contains {target!r}, "
            f"or set target in config.yaml."
        )


def filter_csv(src: Path, dst: Path, drop_names: list[str]) -> None:
    drop = set(drop_names)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="") as handle, dst.open("w", newline="") as out:
        reader = csv.reader(handle)
        writer = csv.writer(out)
        header = next(reader)
        keep = [i for i, name in enumerate(header) if name not in drop]
        writer.writerow([header[i] for i in keep])
        for row in reader:
            writer.writerow([row[i] for i in keep if i < len(row)])


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        # Fall back to a byte copy when symlinks are unavailable.
        dst.write_bytes(src.read_bytes())


def encoder_count(header: list[str]) -> int:
    return sum(1 for name in header if name != "date")


def prepare_one(
    dataset: str,
    source: Path,
    dest_dir: Path,
    target: str,
    drop_rain: bool,
    allow_channel_mismatch: bool,
) -> dict:
    dest = dest_dir / DATA_FILE[dataset]
    if source.resolve() == dest.resolve():
        raise SystemExit(
            f"{source} is already the prepared path. Point LUR_DATA_ROOT at the "
            f"original dataset directory, not {dest_dir}."
        )
    header = read_header(source)
    require_columns(header, source, target)
    raw_enc = encoder_count(header)
    dropped: list[str] = []
    note = ""
    if dataset == "weather" and drop_rain:
        dropped = [name for name in header if is_rain_column(name)]
        if dropped:
            filter_csv(source, dest, dropped)
            prepared_header = read_header(dest)
        else:
            link_or_copy(source, dest)
            prepared_header = header
            if raw_enc == 21:
                preview = ", ".join(header)
                raise SystemExit(
                    "Weather has 21 value columns but none matched rain / raining "
                    "(expected names like 'rain (mm)' and 'raining (s)'). "
                    "Refusing to train on an unrecognized 21-column weather file.\n"
                    f"Header: {preview}"
                )
            note = (
                "No rain / raining column was present. The file was used as-is. "
                "This is a data-quality check, not a method. Confirm the source "
                "already removed those series."
            )
            print(f"[weather] WARNING: {note}")
        enc_in = encoder_count(prepared_header)
        if dropped and enc_in != raw_enc - len(dropped):
            raise SystemExit(
                f"Weather column drop failed: raw {raw_enc}, dropped {dropped}, left {enc_in}."
            )
    else:
        link_or_copy(source, dest)
        prepared_header = header
        enc_in = raw_enc
        if dataset == "weather" and not drop_rain:
            note = "weather_drop_rain is false. Rain columns were kept. This is not the pre-registered protocol."

    if dataset == "weather" and drop_rain and not allow_channel_mismatch and enc_in != 19:
        raise SystemExit(
            f"Weather has {enc_in} value columns after the rain control; expected 19 "
            f"(21 TSLib columns minus rain and raining). Dropped: {dropped or 'none'}."
        )

    expected = EXPECTED_ENCODERS.get(dataset)
    if expected is not None and enc_in != expected and not allow_channel_mismatch:
        raise SystemExit(
            f"{dataset} has {enc_in} value columns; the TSLib file this pilot "
            f"expects has {expected}. Pass --allow-channel-mismatch only if this "
            f"file is intentional."
        )
    if "date" not in prepared_header or target not in prepared_header:
        raise SystemExit(
            f"Prepared {dataset} CSV is missing date or {target}. Dropped: {dropped}."
        )
    record = {
        "dataset": dataset,
        "source": str(source.resolve()),
        "prepared": str(dest.resolve()),
        "data_path": DATA_FILE[dataset],
        "data_key": DATA_KEY[dataset],
        "target": target,
        "freq": "h",
        "raw_enc_in": raw_enc,
        "enc_in": enc_in,
        "columns_dropped": dropped,
        "drop_is_method": False,
        "note": note,
    }
    if dataset == "weather":
        if dropped:
            record["control"] = (
                "Dropped rain / raining columns as a data-quality control before "
                "training. This is not a model, loss, or architecture."
            )
        else:
            record["control"] = (
                "Rain / raining column check ran before training and removed nothing. "
                "This is not a model, loss, or architecture. See note."
            )
    return record


def write_manifest(payload: dict) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_manifest() -> dict:
    path = manifest_path()
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run `python prepare_data.py` before training or diagnosis."
        )
    return json.loads(path.read_text())


def itransformer_arch(dataset: str, horizon: int) -> dict:
    """Hyperparameters from the upstream iTransformer shells at this horizon.

    Traffic is not in that repo. Depth and width are the Traffic script in
    this reproduction's iTransformer/official tree. ECL and weather keep the
    same width at every horizon, matching their upstream shells.
    """
    if dataset == "electricity":
        return {"e_layers": 3, "d_model": 512, "d_ff": 512, "lr": 0.0005, "batch_size": 16}
    if dataset == "weather":
        # weather.sh does not pass --batch_size; run.py defaults to 32.
        return {"e_layers": 3, "d_model": 512, "d_ff": 512, "lr": 0.0001, "batch_size": None}
    if dataset == "traffic":
        return {"e_layers": 4, "d_model": 512, "d_ff": 512, "lr": 0.001, "batch_size": 16}
    if dataset == "ETTh1":
        width = 256 if horizon in (96, 192) else 512
        return {"e_layers": 2, "d_model": width, "d_ff": width, "lr": 0.0001, "batch_size": None}
    if dataset in ("ETTh2", "ETTm1", "ETTm2"):
        return {"e_layers": 2, "d_model": 128, "d_ff": 128, "lr": 0.0001, "batch_size": None}
    raise SystemExit(f"No iTransformer hyperparameters for {dataset}.")


def dlinear_arch(dataset: str) -> dict:
    """Hyperparameters from the upstream DLinear shells.

    The DLinear model does not use d_model, but run_longExp.py puts the
    parser defaults (d_model 512, d_ff 2048, e_layers 2) into the checkpoint
    directory name. Those defaults are kept here so the setting string matches.
    """
    common = {"e_layers": 2, "d_model": 512, "d_ff": 2048}
    table = {
        "electricity": {"lr": 0.001, "batch_size": 16},
        "weather": {"lr": 0.0001, "batch_size": 16},
        # No upstream traffic shell. ECL learning rate and batch size, enc_in 862.
        "traffic": {"lr": 0.001, "batch_size": 16},
        "ETTh1": {"lr": 0.005, "batch_size": 32},
        "ETTh2": {"lr": 0.05, "batch_size": 32},
        "ETTm1": {"lr": 0.0001, "batch_size": 8},
        "ETTm2": {"lr": 0.001, "batch_size": 32},
    }
    if dataset not in table:
        raise SystemExit(f"No DLinear hyperparameters for {dataset}.")
    return {**common, **table[dataset]}


@dataclass
class RunSpec:
    dataset: str
    model: str
    horizon: int
    seq_len: int
    label_len: int
    enc_in: int
    root_path: str
    data_path: str
    data_key: str
    target: str
    freq: str
    model_id: str
    e_layers: int
    d_model: int
    d_ff: int
    learning_rate: float
    batch_size: int | None
    checkpoints: str
    epochs: int
    patience: int
    num_workers: int
    n_heads: int = 8
    d_layers: int = 1
    factor: int = 1
    embed: str = "timeF"
    distil: bool = True
    des: str = "Exp"
    features: str = "M"
    class_strategy: str = "projection"
    lradj: str = "type1"
    use_ps_loss: int = 0
    ps_lambda: float = 0.0
    patch_len_threshold: int = 24
    itr: int = 1
    gpu: int = 0
    dropout: float = 0.1

    def setting(self, ii: int = 0) -> str:
        return setting_from_fields(self, ii=ii)

    def argv(self, is_training: int) -> list[str]:
        args = [
            "--is_training", str(is_training),
            "--root_path", self.root_path,
            "--data_path", self.data_path,
            "--model_id", self.model_id,
            "--model", self.model,
            "--data", self.data_key,
            "--features", self.features,
            "--target", self.target,
            "--freq", self.freq,
            "--checkpoints", self.checkpoints,
            "--seq_len", str(self.seq_len),
            "--label_len", str(self.label_len),
            "--pred_len", str(self.horizon),
            "--enc_in", str(self.enc_in),
            "--des", self.des,
            "--learning_rate", _format_float(self.learning_rate),
            "--lradj", self.lradj,
            "--patience", str(self.patience),
            "--train_epochs", str(self.epochs),
            "--use_ps_loss", str(self.use_ps_loss),
            "--ps_lambda", _format_float(self.ps_lambda),
            "--patch_len_threshold", str(self.patch_len_threshold),
            "--itr", str(self.itr),
            "--num_workers", str(self.num_workers),
            "--gpu", "0",
        ]
        if self.model == "iTransformer":
            args.extend(
                [
                    "--e_layers", str(self.e_layers),
                    "--dec_in", str(self.enc_in),
                    "--c_out", str(self.enc_in),
                    "--d_model", str(self.d_model),
                    "--d_ff", str(self.d_ff),
                ]
            )
        if self.batch_size is not None:
            # Insert next to the other optimization flags. Order does not
            # affect the official parser.
            args.extend(["--batch_size", str(self.batch_size)])
        return args


def _format_float(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def setting_from_fields(spec: RunSpec, ii: int = 0) -> str:
    """Checkpoint directory name written by the official trainers.

    The format strings are copied from TS_Predictability commit
    3bd624602c95506b6f9beedfc43c041fdb6afb36:

    - iTransformer/run.py (the `_ft` label does not line up with `features`;
      the official file passes seq_len in that slot)
    - DLinear/run_longExp.py
    """
    if spec.model == "iTransformer":
        return (
            "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_psloss{}_{}_{}"
        ).format(
            spec.model_id,
            spec.model,
            spec.data_key,
            spec.features,
            spec.seq_len,
            spec.label_len,
            spec.horizon,
            spec.d_model,
            spec.n_heads,
            spec.e_layers,
            spec.d_layers,
            spec.d_ff,
            spec.factor,
            spec.embed,
            spec.distil,
            spec.des,
            spec.use_ps_loss,
            spec.class_strategy,
            ii,
        )
    if spec.model == "DLinear":
        return (
            "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_psloss{}_{}_{}"
        ).format(
            spec.model_id,
            spec.model,
            spec.data_key,
            spec.features,
            spec.seq_len,
            spec.label_len,
            spec.horizon,
            spec.d_model,
            spec.n_heads,
            spec.e_layers,
            spec.d_layers,
            spec.d_ff,
            spec.factor,
            spec.embed,
            spec.distil,
            spec.use_ps_loss,
            spec.des,
            ii,
        )
    raise SystemExit(f"No setting format for model {spec.model}.")


def make_spec(dataset: str, model: str, manifest: dict, cfg: dict, args: argparse.Namespace) -> RunSpec:
    records = manifest["datasets"]
    if dataset not in records:
        raise SystemExit(
            f"{dataset} is not in {manifest_path()}. Re-run prepare_data.py with this dataset."
        )
    record = records[dataset]
    horizon = args.horizon if args.horizon is not None else int(manifest.get("horizon", cfg["horizon"]))
    if horizon != int(manifest.get("horizon")):
        raise SystemExit(
            f"Requested horizon {horizon} but the prepared manifest has horizon "
            f"{manifest.get('horizon')}. Re-run prepare_data.py --horizon {horizon}."
        )
    if model == "iTransformer":
        arch = itransformer_arch(dataset, horizon)
        dropout = 0.1
    elif model == "DLinear":
        arch = dlinear_arch(dataset)
        dropout = 0.05
    else:
        raise SystemExit(f"Unknown model {model}.")
    train_cfg = cfg.get("train") or {}
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("patience", 3))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))
    batch_size = args.batch_size if args.batch_size is not None else arch["batch_size"]
    gpu = args.gpu if args.gpu is not None else int(cfg.get("gpu", 0))
    prefix = MODEL_ID_PREFIX[(dataset, model)]
    checkpoints = work_dir() / "checkpoints" / model
    return RunSpec(
        dataset=dataset,
        model=model,
        horizon=horizon,
        seq_len=horizon,
        label_len=int(cfg.get("label_len", 48)),
        enc_in=int(record["enc_in"]),
        root_path=str(Path(record["prepared"]).parent) + os.sep,
        data_path=record["data_path"],
        data_key=record["data_key"],
        target=record["target"],
        freq=record["freq"],
        model_id=f"{prefix}_{horizon}_{horizon}",
        e_layers=int(arch["e_layers"]),
        d_model=int(arch["d_model"]),
        d_ff=int(arch["d_ff"]),
        learning_rate=float(arch["lr"]),
        batch_size=None if batch_size is None else int(batch_size),
        checkpoints=str(checkpoints.resolve()),
        epochs=epochs,
        patience=patience,
        num_workers=num_workers,
        use_ps_loss=int(train_cfg.get("use_ps_loss", 0)),
        dropout=dropout,
        gpu=gpu,
    )


def script_for(model: str) -> str:
    if model == "iTransformer":
        return "run.py"
    if model == "DLinear":
        return "run_longExp.py"
    raise SystemExit(f"Unknown model {model}.")


def model_dir(model: str) -> Path:
    return upstream_dir() / model


def require_upstream() -> Path:
    root = upstream_dir()
    marker = root / "README.md"
    if not marker.is_file():
        raise SystemExit(
            f"Official code is not at {root}. From SHEAF/lur_pilot run: bash setup.sh"
        )
    head_file = root / ".git" / "HEAD"
    # A detached checkout stores the SHA in HEAD.
    sha = ""
    if head_file.is_file():
        text = head_file.read_text().strip()
        if text.startswith("ref:"):
            ref = root / ".git" / text.split(" ", 1)[1]
            if ref.is_file():
                sha = ref.read_text().strip()
        else:
            sha = text
    if sha and sha != UPSTREAM_COMMIT:
        raise SystemExit(
            f"Upstream checkout is {sha}, expected {UPSTREAM_COMMIT}. Re-run bash setup.sh."
        )
    return root


def weighted_lur(xr: list[float], weights: list[float]) -> float:
    num = 0.0
    den = 0.0
    for value, weight in zip(xr, weights):
        if value != value or weight != weight:  # NaN
            continue
        num += weight * value
        den += weight
    if den == 0.0:
        return float("nan")
    return num / den


def aggregate_bands(per_channel: list[dict]) -> list[dict]:
    """Mean of official per-channel XR_mean and weight_mean.

    per_channel items are {"channel": int, "rows": [{"band", "XR_mean", "weight_mean"}, ...]}.
    """
    bands: list[str] = []
    for item in per_channel:
        for row in item["rows"]:
            if row["band"] not in bands:
                bands.append(row["band"])
    aggregated = []
    for band in bands:
        xr_values = []
        weight_values = []
        for item in per_channel:
            match = next((row for row in item["rows"] if row["band"] == band), None)
            if match is None:
                continue
            xr_values.append(float(match["XR_mean"]))
            weight_values.append(float(match["weight_mean"]))
        aggregated.append(
            {
                "band": band,
                "lur_mean": nanmean(xr_values),
                "weight_mean": nanmean(weight_values),
                "n_channels": len(xr_values),
            }
        )
    return aggregated


def nanmean(values: list[float]) -> float:
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def material_low_bands(bands: list[dict], xr_below: float, min_weight: float) -> list[dict]:
    found = []
    for row in bands:
        lur = row["lur_mean"]
        weight = row["weight_mean"]
        if lur == lur and weight == weight and lur < xr_below and weight >= min_weight:
            found.append(row)
    return found


def kill_verdict(
    by_dataset: dict[str, list[dict]],
    protocol_ok: bool,
    xr_below: float,
    min_weight: float,
) -> dict:
    """Pre-registered stop rule.

    `by_dataset` maps a dataset name to aggregated band rows for the kill-rule
    model (iTransformer). Missing electricity or traffic => incomplete.
    """
    required = list(PREREGISTERED["kill_datasets"])
    per = {}
    missing = [name for name in required if name not in by_dataset]
    for name, bands in by_dataset.items():
        hits = material_low_bands(bands, xr_below, min_weight)
        per[name] = {
            "material_low_bands": hits,
            "n_bands": len(bands),
        }
    if not protocol_ok:
        verdict = "not_preregistered"
        text = (
            "This invocation changed a pre-registered setting (horizon, datasets, "
            "channels, bands, sample cap, Welch parameters, rain-column control, "
            "or the kill thresholds). The stop rule is not answered."
        )
    elif missing:
        verdict = "incomplete"
        text = (
            "The stop rule needs iTransformer band LUR on both Electricity and Traffic. "
            f"Missing: {', '.join(missing)}."
        )
    else:
        decisive = []
        for name in required:
            decisive.extend(per[name]["material_low_bands"])
        weather_hits = per.get("weather", {}).get("material_low_bands", [])
        if not decisive:
            verdict = "stop"
            text = (
                "No material low-LUR band on Electricity or Traffic. Stop. "
                "Do not add a residual module. "
            )
            if weather_hits:
                text += "Weather has a low-LUR band; that is not sufficient."
            else:
                text += "Weather does not change this decision."
        else:
            verdict = "not_stopped"
            text = (
                "A material low-LUR band is present on Electricity and/or Traffic. "
                "This package does not train a residual head. Phase 1 is out of scope."
            )
    return {
        "verdict": verdict,
        "text": text,
        "datasets": per,
        "xr_below": xr_below,
        "min_energy_weight": min_weight,
        "preregistered_protocol": protocol_ok,
    }


def protocol_is_preregistered(cfg_view: dict) -> bool:
    keys = (
        "horizon",
        "lur_bands",
        "lur_max_samples",
        "lur_channels",
        "lur_include_dc",
        "welch_win_frac",
        "welch_overlap",
        "alpha_boundary",
        "xr_below",
        "min_energy_weight",
        "weather_drop_rain",
    )
    for key in keys:
        if cfg_view[key] != PREREGISTERED[key]:
            return False
    if tuple(cfg_view["datasets"]) != tuple(PREREGISTERED["datasets"]):
        return False
    if tuple(cfg_view["kill_datasets"]) != tuple(PREREGISTERED["kill_datasets"]):
        return False
    if cfg_view["kill_model"] != PREREGISTERED["kill_model"]:
        return False
    return True
