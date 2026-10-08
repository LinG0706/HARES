from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hmoe_metrics import hmoe_report_from_predictions, hmoe_summary_columns, hmoe_test_loss_from_predictions

TARGETS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]

def compute_metrics(df: pd.DataFrame, method: str, metric_batch_size: int = 64):
    row = {"method": method, "n_samples": int(len(df)), "n_kernels": int(df["kernel"].nunique())}
    for target in TARGETS:
        err = df[f"y_pred_{target}"] - df[f"y_true_{target}"]
        row[f"{target}_mse"] = float(np.mean(err ** 2))
        row[f"{target}_rmse"] = float(math.sqrt(row[f"{target}_mse"]))
        row[f"{target}_mae"] = float(np.mean(np.abs(err)))
        row[f"{target}_bias"] = float(np.mean(err))
    row["overall_rmse"] = float(math.sqrt(np.mean([row[f"{target}_mse"] for target in TARGETS])))
    per_kernel = []
    for kernel, group in df.groupby("kernel", sort=True):
        item = {"method": method, "kernel": kernel, "n_samples": int(len(group))}
        for target in TARGETS:
            err = group[f"y_pred_{target}"] - group[f"y_true_{target}"]
            item[f"{target}_rmse"] = float(math.sqrt(np.mean(err ** 2)))
            item[f"{target}_mae"] = float(np.mean(np.abs(err)))
            item[f"{target}_bias"] = float(np.mean(err))
        per_kernel.append(item)
    per_kernel_df = pd.DataFrame(per_kernel)
    row["macro_kernel_perf_rmse"] = float(per_kernel_df["perf_rmse"].mean())
    row["macro_kernel_perf_mae"] = float(per_kernel_df["perf_mae"].mean())
    row.update(hmoe_summary_columns(df))
    row.update(hmoe_test_loss_from_predictions(df, batch_size=metric_batch_size))
    return row, per_kernel_df
