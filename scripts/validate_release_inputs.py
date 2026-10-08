from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TARGET_KERNELS = {
    "fdtd-2d-large",
    "gemm-p",
    "gemver-medium",
    "jacobi-2d",
    "syr2k",
    "trmm-opt",
}
MODELS = ["harp_hmoe_support_paired", "graphgps", "exphormer", "polynormer", "gatedgcn"]
EXTERNAL_SOTA = ["sgformer"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_prediction(path: Path, expected_rows: int, expected_ids: set[str] | None = None) -> tuple[pd.DataFrame, set[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"sample_id": str})
    if len(frame) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate sample_id")
    ids = set(frame["sample_id"])
    if expected_ids is not None and ids != expected_ids:
        raise ValueError(f"{path}: sample_id set differs from the first candidate")
    if set(frame["kernel"].astype(str)) != TARGET_KERNELS:
        raise ValueError(f"{path}: target kernel set differs from the final protocol")
    counts = frame.groupby("kernel").size().to_dict()
    if expected_rows == 300 and any(int(counts.get(kernel, 0)) != 50 for kernel in TARGET_KERNELS):
        raise ValueError(f"{path}: support is not 50 samples per kernel: {counts}")
    return frame, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the portable HARES prediction artifact bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", default="1 2 3 4 5")
    args = parser.parse_args()
    root = args.root.resolve()
    seeds = [int(item) for item in args.seeds.replace(",", " ").split()]
    checksums_path = root / "artifacts" / "input_checksums.csv"
    checksums = pd.read_csv(checksums_path, dtype={"path": str})
    checksum_by_path = {
        str(row.path).replace("\\", "/"): row for row in checksums.itertuples(index=False)
    }

    for seed in seeds:
        hmoe = root / "artifacts" / "inputs" / "hmoe" / f"seed{seed}" / "target_finetune"
        stage5e = root / "artifacts" / "inputs" / "stage5e" / f"seed{seed}"
        support_frame, support_ids = check_prediction(hmoe / "target_support_predictions.csv", 300)
        query_frame, query_ids = check_prediction(hmoe / "target_query_predictions.csv", 837)

        manifest_path = root / "artifacts" / "protocol" / f"manifest_kmeans_hmoe_source_exact_seed{seed}.csv"
        manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
        if len(manifest) != 10553:
            raise ValueError(f"{manifest_path}: expected 10553 rows, got {len(manifest)}")
        target = manifest[manifest["dataset_role"].astype(str) == "target"]
        if len(target) != 1137:
            raise ValueError(f"{manifest_path}: expected 1137 target rows, got {len(target)}")
        manifest_support = set(target.loc[target["split"].astype(str) == "support", "sample_id"])
        manifest_query = set(target.loc[target["split"].astype(str) == "query", "sample_id"])
        if manifest_support != support_ids or manifest_query != query_ids:
            raise ValueError(f"{manifest_path}: manifest IDs differ from prediction IDs")
        if any(int(count) != 50 for count in target[target["split"].astype(str) == "support"].groupby("kernel").size()):
            raise ValueError(f"{manifest_path}: support is not 50 per target kernel")

        for sota_name in EXTERNAL_SOTA:
            sota_dir = root / "artifacts" / "inputs" / "sota" / sota_name / f"seed{seed}"
            for split, expected_rows, expected_ids in (
                ("support", 300, support_ids),
                ("query", 837, query_ids),
            ):
                prediction = sota_dir / f"target_{split}_predictions.csv"
                check_prediction(prediction, expected_rows, expected_ids)
                checksum_key = prediction.relative_to(root / "artifacts").as_posix()
                checksum_row = checksum_by_path.get(checksum_key)
                if checksum_row is None:
                    raise ValueError(f"{checksums_path}: missing checksum for {checksum_key}")
                observed_hash = sha256(prediction)
                if observed_hash != str(checksum_row.sha256):
                    raise ValueError(f"{prediction}: SHA-256 mismatch against {checksums_path}")
                if int(checksum_row.rows) != expected_rows:
                    raise ValueError(f"{checksums_path}: row-count metadata mismatch for {checksum_key}")

        lock_path = stage5e / "selection_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("query_labels_used_for_selection") is not False:
            raise ValueError(f"{lock_path}: query-label selection flag is not false")
        candidate_items = lock.get("candidate_hashes", [])
        for model in MODELS:
            for split, expected_rows, expected_ids in (("support", 300, support_ids), ("query", 837, query_ids)):
                item = next((row for row in candidate_items if row.get("model") == model and row.get("split") == split), None)
                if item is None:
                    raise ValueError(f"{lock_path}: missing {model}/{split} candidate")
                candidate = root / Path(item["path"])
                check_prediction(candidate, expected_rows, expected_ids)
                observed_hash = sha256(candidate)
                if observed_hash != item.get("sha256"):
                    raise ValueError(f"{candidate}: SHA-256 mismatch")

        if not (stage5e / "stage5e_convex_weights.csv").exists():
            raise FileNotFoundError(stage5e / "stage5e_convex_weights.csv")
        if not (stage5e / "target_query_predictions.csv").exists():
            raise FileNotFoundError(stage5e / "target_query_predictions.csv")
        print(f"seed{seed}: passed ({len(support_frame)} support, {len(query_ids)} query, SGFormer IDs and hashes checked)")

    print("All HARES release inputs passed the protocol and hash checks.")


if __name__ == "__main__":
    main()
