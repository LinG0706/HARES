from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]


def mse_row(frame: pd.DataFrame, method: str, seed: int) -> dict:
    row = {"method": method, "seed": seed}
    for target in TARGETS:
        errors = (
            frame[f"y_pred_{target}"].to_numpy(dtype=np.float64)
            - frame[f"y_true_{target}"].to_numpy(dtype=np.float64)
        )
        row[f"{target}_mse"] = float(np.mean(errors * errors))
    row["total_mse"] = sum(row[f"{target}_mse"] for target in TARGETS)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's sample-weighted QoR metric.")
    parser.add_argument("--hmoe-root", type=Path, default=Path("artifacts/inputs/hmoe"))
    parser.add_argument("--stage5e-root", type=Path, default=Path("artifacts/inputs/stage5e"))
    parser.add_argument("--candidate-root", type=Path, default=Path("artifacts/inputs/candidates"))
    parser.add_argument("--hares-root", type=Path, default=Path("outputs/stage5i"))
    parser.add_argument("--seeds", default="1 2 3 4 5")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.replace(",", " ").split()]
    records = []
    for seed in seeds:
        sources = {
            "HARP": args.candidate_root / f"seed{seed}" / "harp_hmoe_support_paired" / "target_query_predictions.csv",
            "HMoE": args.hmoe_root / f"seed{seed}" / "target_finetune" / "target_query_predictions.csv",
            "Unshrunk pool": args.stage5e_root / f"seed{seed}" / "target_query_predictions.csv",
            "HARES": args.hares_root / f"seed{seed}" / "target_query_predictions.csv",
        }
        for method, path in sources.items():
            frame = pd.read_csv(path, dtype={"sample_id": str})
            if len(frame) != 837 or frame["sample_id"].duplicated().any():
                raise ValueError(f"{path}: expected 837 unique query predictions")
            records.append(mse_row(frame, method, seed))

    per_seed = pd.DataFrame(records)
    columns = [f"{target}_mse" for target in TARGETS] + ["total_mse"]
    summary = per_seed.groupby("method", sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()

    print("Sample-weighted QoR MSE: mean ± sample std over seeds")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
