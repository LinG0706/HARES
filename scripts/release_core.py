from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]
KEY_TOKEN_RE = re.compile(r"__(?P<kind>PARA|PIPE|TILE)__L(?P<loop>\d+)-(?P<value>[^.]+)")
META_COLUMNS = [
    "sample_id", "kernel", "gname", "key", "dataset_role", "split",
    "support_query", "cache_split", "file_name", "actual_perf",
    "seed", "k", "selection",
]


def parse_candidate(spec: str):
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        return name, Path(raw_path)
    path = Path(spec)
    return path.name, path


def read_prediction_dir(path: Path, model_name: str, split: str):
    csv_path = path / f"target_{split}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    frame["model"] = model_name
    return frame


def align_prediction_frames(frames):
    if not frames:
        raise ValueError("no prediction frames")
    base_ids = set(frames[0]["sample_id"])
    base = frames[0][META_COLUMNS + [f"y_true_{target}" for target in TARGETS]].copy()
    aligned = []
    for frame in frames:
        ids = set(frame["sample_id"])
        if ids != base_ids:
            missing = sorted(base_ids - ids)[:5]
            extra = sorted(ids - base_ids)[:5]
            raise ValueError(f"prediction sample_id sets differ: missing={missing}, extra={extra}")
        aligned.append(base[["sample_id"]].merge(frame, on="sample_id", how="left", validate="one_to_one"))
    return aligned


def load_candidates(candidate_specs, split: str):
    return align_prediction_frames([read_prediction_dir(path, name, split) for name, path in map(parse_candidate, candidate_specs)])


def model_names_from_frames(frames):
    return [str(frame["model"].iloc[0]) for frame in frames]


def parse_float_grid(raw: str):
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("empty float grid")
    return values


def parse_int_grid(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("empty int grid")
    return values


def prediction_matrix(frames, target: str, indices=None):
    columns = []
    for frame in frames:
        values = frame[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        columns.append(values if indices is None else values[indices])
    return np.column_stack(columns)


def true_values(frame, target: str, indices=None):
    values = frame[f"y_true_{target}"].to_numpy(dtype=np.float64)
    return values if indices is None else values[indices]


def predict_with_models(frames, models, scope: str, model_label: str):
    base = frames[0].copy()
    for target in TARGETS:
        predictions = np.zeros(len(base), dtype=np.float64)
        if scope == "global":
            item = models[("global", target)]
            predictions = prediction_matrix(frames, target) @ item["weights"] + item["bias"]
        elif scope == "per_kernel":
            for kernel, group in base.groupby("kernel", sort=True):
                item = models[(kernel, target)]
                indices = group.index.to_numpy()
                predictions[indices] = prediction_matrix(frames, target, indices) @ item["weights"] + item["bias"]
        else:
            raise ValueError(f"unsupported scope: {scope}")
        base[f"y_pred_{target}"] = predictions
    base["model"] = model_label
    return base.reset_index(drop=True)


def write_weights(models, model_names, out_path: Path):
    rows = []
    for (scope_key, target), item in models.items():
        row = {"scope_key": scope_key, "target": target, "bias": item["bias"]}
        row.update({f"weight::{name}": float(weight) for name, weight in zip(model_names, item["weights"])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_design_key(key: str):
    values = {}
    for match in KEY_TOKEN_RE.finditer(str(key)):
        name = f"design::{match.group('kind')}::L{match.group('loop')}"
        raw = match.group("value")
        if raw == "off":
            number, is_off, is_flatten, is_na = 0.0, 1.0, 0.0, 0.0
        elif raw == "flatten":
            number, is_off, is_flatten, is_na = 1.0, 0.0, 1.0, 0.0
        elif raw == "NA":
            number, is_off, is_flatten, is_na = 0.0, 0.0, 0.0, 1.0
        else:
            try:
                number = float(raw)
            except ValueError:
                number = 0.0
            is_off, is_flatten, is_na = 0.0, 0.0, 0.0
        values[f"{name}::num"] = number
        values[f"{name}::is_off"] = is_off
        values[f"{name}::is_flatten"] = is_flatten
        values[f"{name}::is_na"] = is_na
    return values


def split_support_indices(base: pd.DataFrame, val_per_kernel: int, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for _, group in base.groupby("kernel", sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_val = min(val_per_kernel, max(len(indices) - 1, 1))
        val_idx.extend(indices[:n_val].tolist())
        train_idx.extend(indices[n_val:].tolist())
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def write_json(path: Path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
