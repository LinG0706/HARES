"""Shared checks for the fixed query set and five-target evaluation space."""
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]
MODEL_NAMES = {
    "harp": "HARP", "graphgps": "GraphGPS", "exphormer": "Exphormer",
    "polynormer": "Polynormer", "gatedgcn": "GatedGCN",
    "gnndse": "GNN-DSE", "ironman": "IronMan-Pro",
}


def check_prediction(path: Path, reference: pd.DataFrame, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = ["sample_id", "kernel"] + [f"{prefix}_{t}" for t in TARGETS for prefix in ("y_true", "y_pred")]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate sample_id")
    if set(frame["sample_id"]) != set(reference["sample_id"]):
        raise ValueError(f"{path}: IDs differ from seed{seed} protocol")
    frame = frame.set_index("sample_id").loc[reference["sample_id"]].reset_index()
    if not np.array_equal(frame["kernel"].astype(str).to_numpy(), reference["kernel"].astype(str).to_numpy()):
        raise ValueError(f"{path}: kernel mismatch")
    if "seed" in frame and not (pd.to_numeric(frame["seed"]) == seed).all():
        raise ValueError(f"{path}: seed metadata mismatch")
    for target in TARGETS:
        true = frame[f"y_true_{target}"].to_numpy(dtype=np.float64)
        pred = frame[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        if not np.isfinite(true).all() or not np.isfinite(pred).all():
            raise ValueError(f"{path}: non-finite {target}")
        expected = reference[target].to_numpy(dtype=np.float64)
        if not np.allclose(true, expected, rtol=0, atol=1e-6):
            raise ValueError(f"{path}: {target} labels differ from protocol")
    return frame


def manifest_split(root: Path, seed: int, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(root / "artifacts/protocol" / f"manifest_kmeans_hmoe_source_exact_seed{seed}.csv", dtype={"sample_id": str})
    if len(manifest) != 10553 or manifest["sample_id"].duplicated().any():
        raise ValueError(f"seed{seed}: invalid full manifest")
    selected = manifest[(manifest["dataset_role"] == "target") & (manifest["split"] == split)].copy()
    expected_n = {"support": 300, "query": 837}[split]
    if len(selected) != expected_n:
        raise ValueError(f"seed{seed}: expected {expected_n} {split} records")
    if split == "support" and not (selected.groupby("kernel").size() == 50).all():
        raise ValueError(f"seed{seed}: support must have 50 designs per kernel")
    # Accept hyphenated resource names as well as the released underscore names.
    selected = selected.rename(columns={f"util-{r}": f"util_{r}" for r in ("LUT", "FF", "DSP", "BRAM")})
    return selected.reset_index(drop=True)
