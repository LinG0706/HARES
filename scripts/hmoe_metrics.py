from pathlib import Path
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


TARGETS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]
HMOE_TARGET_LABELS = {
    "perf": "perf",
    "util_LUT": "util-LUT",
    "util_FF": "util-FF",
    "util_DSP": "util-DSP",
    "util_BRAM": "util-BRAM",
}


def _mse(true_values, pred_values):
    return float(mean_squared_error(true_values, pred_values))


def _rmse(true_values, pred_values):
    return float(np.sqrt(_mse(true_values, pred_values)))


def hmoe_report_from_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror HMOE train.py::_report_rmse_etc on an exported prediction CSV.

    HMOE reports one row per target and a final "tot/avg" row.  Despite the
    name, "tot/avg" sums mape/rmse/mse/mae/max_err across targets and averages
    only Kendall tau.
    """

    data = defaultdict(list)
    totals = {
        "mape": 0.0,
        "rmse": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "max_err": 0.0,
        "tau": 0.0,
    }

    for target in TARGETS:
        true_values = df[f"y_true_{target}"].to_numpy(dtype=np.float64)
        pred_values = df[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        true_rank = rankdata(true_values)
        pred_rank = rankdata(pred_values)
        tau = kendalltau(true_rank, pred_rank)[0]
        if pd.isna(tau):
            tau = np.nan

        row = {
            "target": HMOE_TARGET_LABELS[target],
            "n_samples": int(len(true_values)),
            "mape": float(mean_absolute_percentage_error(true_values, pred_values)),
            "rmse": _rmse(true_values, pred_values),
            "mse": _mse(true_values, pred_values),
            "mae": float(mean_absolute_error(true_values, pred_values)),
            "max_err": float(max_error(true_values, pred_values)),
            "tau": float(tau) if not pd.isna(tau) else np.nan,
        }

        for key, value in row.items():
            data[key].append(value)
        for key in totals:
            if key == "tau" and pd.isna(row[key]):
                continue
            totals[key] += row[key]

    n_targets = len(TARGETS)
    data["target"].append("tot/avg")
    data["n_samples"].append(int(len(df)))
    data["mape"].append(totals["mape"])
    data["rmse"].append(totals["rmse"])
    data["mse"].append(totals["mse"])
    data["mae"].append(totals["mae"])
    data["max_err"].append(totals["max_err"])
    data["tau"].append(totals["tau"] / n_targets)
    return pd.DataFrame(data)


def hmoe_summary_columns(df: pd.DataFrame) -> dict:
    report = hmoe_report_from_predictions(df)
    perf = report[report["target"] == "perf"].iloc[0]
    total = report[report["target"] == "tot/avg"].iloc[0]
    overall_rmse = float(np.sqrt(float(total["mse"]) / len(TARGETS)))
    return {
        "hmoe_perf_rmse": float(perf["rmse"]),
        "hmoe_perf_mse": float(perf["mse"]),
        "hmoe_perf_mae": float(perf["mae"]),
        "hmoe_perf_tau": None if pd.isna(perf["tau"]) else float(perf["tau"]),
        "hmoe_report_overall_rmse": overall_rmse,
        "hmoe_report_rmse_sum": float(total["rmse"]),
        "hmoe_report_mse_sum": float(total["mse"]),
        "hmoe_report_mae_sum": float(total["mae"]),
        "hmoe_report_tau_avg": None if pd.isna(total["tau"]) else float(total["tau"]),
    }


def hmoe_test_loss_from_predictions(df: pd.DataFrame, batch_size: int = 64) -> dict:
    """Reproduce HMOE train.py::test loss from exported predictions.

    In HMOE, each batch loss is the sum of MSE losses over the five regression
    targets.  The epoch test loss is then the unweighted mean over batches.
    This intentionally gives the final partial batch the same weight as a full
    batch, matching the original project.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if df.empty:
        raise ValueError("prediction dataframe is empty")

    batch_losses = []
    per_target_batch_losses = {target: [] for target in TARGETS}
    for start in range(0, len(df), batch_size):
        group = df.iloc[start : start + batch_size]
        total = 0.0
        for target in TARGETS:
            err = group[f"y_pred_{target}"].to_numpy(dtype=np.float64) - group[f"y_true_{target}"].to_numpy(dtype=np.float64)
            mse = float(np.mean(err ** 2))
            per_target_batch_losses[target].append(mse)
            total += mse
        batch_losses.append(total)

    out = {
        "hmoe_test_loss": float(np.mean(batch_losses)),
        "hmoe_metric_batch_size": int(batch_size),
        "hmoe_num_batches": int(len(batch_losses)),
    }
    for target in TARGETS:
        out[f"hmoe_{target}_loss"] = float(np.mean(per_target_batch_losses[target]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    prediction = Path(args.prediction)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(prediction)
    report = hmoe_report_from_predictions(df)
    report.to_csv(out, index=False)
    print(report.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(hmoe_test_loss_from_predictions(df))
    print(f"wrote HMOE-style report to: {out}")


if __name__ == "__main__":
    main()
