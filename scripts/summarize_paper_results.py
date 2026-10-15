"""Recompute the complete paper table from the five paired query exports."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_protocol import MODEL_NAMES, TARGETS, check_prediction, manifest_split


def mse_row(frame: pd.DataFrame, method: str, seed: int) -> dict:
    row = {"method": method, "seed": seed, "n_query": len(frame)}
    for target in TARGETS:
        error = frame[f"y_pred_{target}"].to_numpy(dtype=np.float64) - frame[f"y_true_{target}"].to_numpy(dtype=np.float64)
        row[f"{target}_mse"] = float(np.mean(error ** 2))
    row["total_mse"] = sum(row[f"{target}_mse"] for target in TARGETS)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hmoe-root", type=Path, default=Path("artifacts/inputs/hmoe"))
    parser.add_argument("--stage5e-root", type=Path, default=Path("artifacts/inputs/stage5e"))
    parser.add_argument("--candidate-root", type=Path, default=Path("artifacts/inputs/candidates"))
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/inputs/baselines"))
    parser.add_argument("--hares-root", type=Path, default=Path("outputs/stage5i"))
    parser.add_argument("--seeds", default="1 2 3 4 5")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    def resolve(path):
        return path if path.is_absolute() else root / path
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds")
    records = []
    for seed in seeds:
        reference = manifest_split(root, seed, "query")
        sources = {
            "HARP": resolve(args.candidate_root) / f"seed{seed}/harp/target_query_predictions.csv",
            "HMoE": resolve(args.hmoe_root) / f"seed{seed}/target_finetune/target_query_predictions.csv",
            "Unshrunk pool": resolve(args.stage5e_root) / f"seed{seed}/target_query_predictions.csv",
            "HARES": resolve(args.hares_root) / f"seed{seed}/target_query_predictions.csv",
        }
        for model in ("gnndse", "ironman"):
            sources[MODEL_NAMES[model]] = resolve(args.baseline_root) / model / f"seed{seed}/target_query_predictions.csv"
        for model in ("graphgps", "exphormer", "polynormer", "gatedgcn"):
            sources[MODEL_NAMES[model]] = resolve(args.candidate_root) / f"seed{seed}" / model / "target_query_predictions.csv"
        for method, path in sources.items():
            records.append(mse_row(check_prediction(path, reference, seed), method, seed))

    per_seed = pd.DataFrame(records)
    columns = [f"{t}_mse" for t in TARGETS] + ["total_mse"]
    summary = per_seed.groupby("method", sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    first = ["HARP", "HMoE", "Unshrunk pool", "HARES"]
    others = summary.drop(index=first).sort_values("total_mse_mean", ascending=False).index.tolist()
    summary = summary.loc[first + others].reset_index()
    summary.insert(1, "n_seeds", len(seeds))
    print("Sample-weighted five-target MSE; mean +/- sample standard deviation (ddof=1)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
    if args.output_dir:
        out = resolve(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(out / "per_seed.csv", index=False)
        summary.to_csv(out / "mean_std.csv", index=False)
        latex = [
            r"\begin{table*}[!t]", r"\centering",
            r"\caption{QoR prediction comparison. Per-target columns report sample-weighted MSE; Total MSE is their sum. Values are mean $\pm$ sample standard deviation over five seeds.}",
            r"\label{tab:main}", r"\renewcommand{\arraystretch}{1}",
            r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
            r"\textbf{Method} & \textbf{Perf. MSE} & \textbf{LUT MSE} & \textbf{FF MSE} & \textbf{DSP MSE} & \textbf{BRAM MSE} & \textbf{Total MSE}\\", r"\midrule",
        ]
        citations = {"HARP": "harp", "HMoE": "hmoe", "GNN-DSE": "gnndse", "IronMan-Pro": "ironman", "GraphGPS": "graphgps", "Exphormer": "exphormer", "Polynormer": "polynormer", "GatedGCN": "gatedgcn"}
        for _, row in summary.iterrows():
            if row["method"] == others[0]:
                latex.append(r"\midrule")
            name = row["method"]
            label = r"\textbf{HARES}" if name == "HARES" else name
            if name in citations:
                label += r"~\cite{" + citations[name] + "}"
            values = []
            for metric in columns:
                value = f"{row[metric + '_mean']:.4f}\\pm{row[metric + '_std']:.4f}"
                if row[metric + "_mean"] == summary[metric + "_mean"].min():
                    value = r"\mathbf{" + value + "}"
                values.append("$" + value + "$")
            latex.append(label + " & " + " & ".join(values) + r"\\")
        latex += [r"\bottomrule", r"\end{tabular}%", "}", r"\vspace{-3mm}", r"\end{table*}"]
        (out / "main_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
