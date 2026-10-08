from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


TARGETS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]


def summarize(run_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for seed in seeds:
        path = run_root / f"seed{seed}" / "support_ft" / "target_query_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        manifest_path = Path(__file__).resolve().parents[2] / f"artifacts/protocol/manifest_kmeans_hmoe_source_exact_seed{seed}.csv"
        manifest = pd.read_csv(manifest_path)
        expected = manifest[(manifest["dataset_role"] == "target") & (manifest["split"] == "query")]
        if len(frame) != len(expected) or set(frame["sample_id"]) != set(expected["sample_id"]):
            raise ValueError(f"seed{seed}: left-out design IDs do not match the HMoE manifest")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"seed{seed}: duplicate prediction IDs")
        indexed = frame.set_index("sample_id")
        expected = expected.set_index("sample_id").loc[indexed.index]
        for target in TARGETS:
            if not np.allclose(indexed[f"y_true_{target}"], expected[target], rtol=0, atol=1e-5):
                raise ValueError(f"seed{seed}: {target} labels do not match the HMoE manifest")
        row = {"seed": seed, "n_query": len(frame)}
        for target in TARGETS:
            error = frame[f"y_pred_{target}"].to_numpy(dtype=float) - frame[f"y_true_{target}"].to_numpy(dtype=float)
            row[f"{target}_mse"] = float(np.mean(error ** 2))
            row[f"{target}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        row["total_mse"] = float(sum(row[f"{target}_mse"] for target in TARGETS))
        rows.append(row)

    per_seed = pd.DataFrame(rows)
    metric_cols = [f"{target}_mse" for target in TARGETS] + ["total_mse"]
    summary = pd.DataFrame({
        "mean": per_seed[metric_cols].mean(),
        "sample_std": per_seed[metric_cols].std(ddof=1),
    })
    payload = {
        "baseline": "SGFormer-RGCN",
        "protocol": "HMoE-aligned adapter",
        "predictor_mode": "shared",
        "seeds": seeds,
        "targets": TARGETS,
        "metric_definition": "Per-target sample MSE on left-out designs; Total MSE is the sum of five target MSEs",
        "original_method_note": "The original Tang implementation uses a different dataset and HLS-report global features. This adapter uses HMoE graph/pragma inputs and the HMoE source/support/query protocol.",
        "per_seed": per_seed.to_dict(orient="records"),
        "summary": summary.reset_index(names="metric").to_dict(orient="records"),
    }
    return per_seed, summary, payload


def fmt(summary: pd.DataFrame, metric: str) -> str:
    return f"{summary.loc[metric, 'mean']:.4f}\\pm{summary.loc[metric, 'sample_std']:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, summary, payload = summarize(args.run_root, args.seeds)
    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    summary.reset_index(names="metric").to_csv(args.output_dir / "mean_std.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metrics = ["perf_mse", "util_LUT_mse", "util_FF_mse", "util_DSP_mse", "util_BRAM_mse", "total_mse"]
    latex = "SGFormer-RGCN$^{\\dagger}$" + " & " + " & ".join(f"${fmt(summary, metric)}$" for metric in metrics) + " \\\\\n"
    (args.output_dir / "table_row.tex").write_text(latex, encoding="utf-8")
    print(latex, end="")
    print(f"Saved summary files under {args.output_dir}")


if __name__ == "__main__":
    main()
