from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hmoe_metrics import hmoe_report_from_predictions
from hmoe_metrics import TARGETS
from release_core import write_json
from release_metrics import compute_metrics


METHOD_ID = "FSS-R4-M1"
METHOD_NAME = "Stage5I-HMoE-target-shrinkage"
STAGE5E_METHOD_NAME = "Stage5E-DMW-AFRC"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_recorded_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    normalized = raw_path.replace("\\", "/")
    marker = "experiments/new4+harp/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = workspace_root() / "experiments" / "new4+harp" / Path(suffix)
        if candidate.exists():
            return candidate
    return path


def load_stage5e_candidate_frames(stage5e_dir: Path, split: str) -> tuple[list[str], list[pd.DataFrame], list[dict]]:
    lock_path = stage5e_dir / "selection_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Stage5E selection lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidate_hashes = lock.get("candidate_hashes")
    if candidate_hashes is None:
        if split == "support":
            candidate_hashes = lock.get("support_candidate_hashes", [])
        elif split == "query":
            candidate_hashes = lock.get("query_candidate_hashes", [])
        else:
            candidate_hashes = []
    entries = [item for item in candidate_hashes if item.get("split", split) == split]
    if not entries:
        raise ValueError(f"Stage5E lock has no candidate hashes for split={split}: {lock_path}")

    model_names = [str(item["model"]) for item in entries]
    frames = []
    hash_rows = []
    base_ids = None
    for item in entries:
        path = resolve_recorded_path(str(item["path"]))
        if not path.exists():
            raise FileNotFoundError(f"Stage5E candidate prediction missing for {item['model']} {split}: {path}")
        digest = sha256_file(path)
        expected = item.get("sha256")
        if expected and digest != expected:
            raise ValueError(f"Stage5E candidate hash mismatch for {path}: {digest} != {expected}")
        frame = pd.read_csv(path).reset_index(drop=True)
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"duplicate sample_id in Stage5E candidate {path}")
        if base_ids is None:
            base_ids = frame["sample_id"].astype(str).tolist()
        elif frame["sample_id"].astype(str).tolist() != base_ids:
            raise ValueError(f"Stage5E candidate row order mismatch for {path}")
        frame["model"] = str(item["model"])
        frames.append(frame)
        hash_rows.append(
            {
                "artifact": f"stage5e_candidate_{split}",
                "model": str(item["model"]),
                "path": str(path),
                "sha256": digest,
            }
        )
    return model_names, frames, hash_rows


def stage5e_models_from_weights(stage5e_dir: Path, model_names: list[str]) -> tuple[dict, str]:
    weight_path = stage5e_dir / "stage5e_convex_weights.csv"
    if not weight_path.exists():
        raise FileNotFoundError(f"Stage5E weights missing: {weight_path}")
    weights = pd.read_csv(weight_path)
    weight_cols = [col for col in weights.columns if col.startswith("weight::")]
    weight_models = [col.split("weight::", 1)[1] for col in weight_cols]
    if weight_models != model_names:
        raise ValueError(f"Stage5E weight/candidate model order mismatch: weights={weight_models}, candidates={model_names}")

    scopes = set(weights["scope_key"].astype(str))
    scope = "global" if scopes == {"global"} else "per_kernel"
    models = {}
    for _, row in weights.iterrows():
        key = (str(row["scope_key"]), str(row["target"]))
        models[key] = {
            "bias": float(row.get("bias", 0.0)),
            "weights": np.asarray([float(row[col]) for col in weight_cols], dtype=np.float64),
        }
    return models, scope


def build_stage5e_prediction_from_locked_weights(stage5e_dir: Path, split: str) -> tuple[pd.DataFrame, list[dict]]:
    model_names, frames, hash_rows = load_stage5e_candidate_frames(stage5e_dir, split)
    models, scope = stage5e_models_from_weights(stage5e_dir, model_names)
    base = frames[0].copy().reset_index(drop=True)
    for target in TARGETS:
        preds = np.zeros(len(base), dtype=np.float64)
        if scope == "global":
            item = models[("global", target)]
            matrix = np.column_stack([frame[f"y_pred_{target}"].to_numpy(dtype=np.float64) for frame in frames])
            preds = matrix @ item["weights"] + item["bias"]
        else:
            for kernel, group in base.groupby("kernel", sort=True):
                item = models[(str(kernel), target)]
                idx = group.index.to_numpy(dtype=int)
                matrix = np.column_stack([frame.loc[idx, f"y_pred_{target}"].to_numpy(dtype=np.float64) for frame in frames])
                preds[idx] = matrix @ item["weights"] + item["bias"]
        base[f"y_pred_{target}"] = preds
        base[f"abs_err_{target}"] = np.abs(preds - base[f"y_true_{target}"].to_numpy(dtype=np.float64))
    base["model"] = STAGE5E_METHOD_NAME
    return base.reset_index(drop=True), hash_rows


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.replace(",", " ").split() if item.strip()]


def parse_floats(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.replace(",", " ").split() if item.strip()})
    if 0.0 not in values:
        values = [0.0, *values]
    if min(values) < 0.0 or max(values) > 1.0:
        raise ValueError(f"lambda grid must be within [0, 1]: {values}")
    return values


def align_pair(anchor: pd.DataFrame, correction: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if anchor["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id in HMoE {split}")
    if correction["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id in Stage5E {split}")
    if set(anchor["sample_id"]) != set(correction["sample_id"]):
        missing = sorted(set(anchor["sample_id"]) - set(correction["sample_id"]))[:5]
        extra = sorted(set(correction["sample_id"]) - set(anchor["sample_id"]))[:5]
        raise ValueError(f"{split} sample_id mismatch: missing={missing}, extra={extra}")
    correction = anchor[["sample_id"]].merge(correction, on="sample_id", how="left", validate="one_to_one")
    for col in ["kernel", "key"]:
        if col in anchor.columns and col in correction.columns:
            if not (anchor[col].astype(str).to_numpy() == correction[col].astype(str).to_numpy()).all():
                raise ValueError(f"{split} metadata mismatch for {col}")
    for target in TARGETS:
        a = anchor[f"y_true_{target}"].to_numpy(dtype=np.float64)
        b = correction[f"y_true_{target}"].to_numpy(dtype=np.float64)
        if not np.allclose(a, b, rtol=0.0, atol=1e-6):
            raise ValueError(f"{split} y_true mismatch for {target}")
    return anchor.reset_index(drop=True), correction.reset_index(drop=True)


def mse_for_target(frame: pd.DataFrame, target: str) -> float:
    err = frame[f"y_pred_{target}"].to_numpy(dtype=np.float64) - frame[f"y_true_{target}"].to_numpy(dtype=np.float64)
    return float(np.mean(err ** 2))


def select_lambdas(
    hmoe_support: pd.DataFrame,
    stage5e_support: pd.DataFrame,
    lambda_grid: list[float],
    min_support_rel_gain: float,
    one_se_tolerance: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    lambdas: dict[str, float] = {}
    rows = []
    for target in TARGETS:
        y_true = hmoe_support[f"y_true_{target}"].to_numpy(dtype=np.float64)
        base = hmoe_support[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        corr = stage5e_support[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        support_mses = []
        for value in lambda_grid:
            pred = base + value * (corr - base)
            support_mses.append(float(np.mean((pred - y_true) ** 2)))
        hmoe_mse = support_mses[lambda_grid.index(0.0)]
        best_mse = min(support_mses)
        best_lambda = lambda_grid[int(np.argmin(support_mses))]
        rel_gain = 0.0 if hmoe_mse <= 0 else (hmoe_mse - best_mse) / hmoe_mse
        if rel_gain < min_support_rel_gain:
            selected = 0.0
            reason = "support_gain_below_threshold"
        else:
            threshold = best_mse * (1.0 + one_se_tolerance)
            eligible = [value for value, mse in zip(lambda_grid, support_mses) if mse <= threshold]
            selected = min(eligible)
            reason = "smallest_lambda_within_tolerance"
        lambdas[target] = float(selected)
        for value, mse in zip(lambda_grid, support_mses):
            rows.append(
                {
                    "target": target,
                    "lambda": float(value),
                    "support_mse": mse,
                    "hmoe_support_mse": hmoe_mse,
                    "best_lambda": float(best_lambda),
                    "best_support_mse": best_mse,
                    "support_rel_gain": float(rel_gain),
                    "selected_lambda": float(selected),
                    "selection_reason": reason,
                }
            )
    return lambdas, pd.DataFrame(rows)


def blend_predictions(hmoe: pd.DataFrame, stage5e: pd.DataFrame, lambdas: dict[str, float], seed: int) -> pd.DataFrame:
    out = hmoe.copy()
    out["model"] = METHOD_NAME
    out["seed"] = seed
    out["method_id"] = METHOD_ID
    for target in TARGETS:
        lam = lambdas[target]
        base = hmoe[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        corr = stage5e[f"y_pred_{target}"].to_numpy(dtype=np.float64)
        pred = base + lam * (corr - base)
        out[f"y_pred_{target}"] = pred
        out[f"abs_err_{target}"] = np.abs(pred - out[f"y_true_{target}"].to_numpy(dtype=np.float64))
    return out


def metric_record(frame: pd.DataFrame, method: str, metric_batch_size: int) -> dict:
    row, _ = compute_metrics(frame, method=method, metric_batch_size=metric_batch_size)
    return row


def method_row(seed: int, method: str, metrics: dict, hmoe_loss: float) -> dict:
    row = {
        "seed": seed,
        "method": method,
        "hmoe_test_loss": float(metrics["hmoe_test_loss"]),
        "perf_rmse": float(metrics["perf_rmse"]),
        "util_LUT_rmse": float(metrics["util_LUT_rmse"]),
        "util_FF_rmse": float(metrics["util_FF_rmse"]),
        "util_DSP_rmse": float(metrics["util_DSP_rmse"]),
        "util_BRAM_rmse": float(metrics["util_BRAM_rmse"]),
        "overall_rmse": float(metrics["overall_rmse"]),
        "hmoe_report_rmse_sum": float(metrics["hmoe_report_rmse_sum"]),
    }
    row["delta_vs_hmoe"] = row["hmoe_test_loss"] - hmoe_loss
    row["relative_change_vs_hmoe"] = row["delta_vs_hmoe"] / hmoe_loss
    row["better_than_hmoe"] = row["delta_vs_hmoe"] < 0
    return row


def write_archive(out_root: Path, archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    for name in [
        "stage5i_per_seed.csv",
        "stage5i_mean_std.csv",
        "RESULT_FILE_SHA256.csv",
        "README.md",
    ]:
        src = out_root / name
        if src.exists():
            shutil.copy2(src, archive_root / name)
    for seed_dir in sorted(out_root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        dst = archive_root / seed_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in [
            "selection_lock.json",
            "stage5i_lambda_search.csv",
            "stage5i_query_summary.csv",
            "stage5i_support_summary.csv",
        ]:
            src = seed_dir / name
            if src.exists():
                shutil.copy2(src, dst / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmoe-root", type=Path, default=Path("experiments/new4+harp/hmoe_paper_exact_baseline"))
    parser.add_argument("--stage5e-root", type=Path, default=Path("experiments/new4+harp/stage5e_paper_exact_seed1_5"))
    parser.add_argument("--out-root", type=Path, default=Path("experiments/new4+harp/stage5i_hmoe_shrinkage_seed1_5"))
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--seeds", default="1 2 3 4 5")
    parser.add_argument("--lambda-grid", default="0 0.05 0.1 0.2 0.3")
    parser.add_argument("--min-support-rel-gain", type=float, default=0.05)
    parser.add_argument("--one-se-tolerance", type=float, default=0.02)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    lambda_grid = parse_floats(args.lambda_grid)
    args.out_root.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    hash_rows = []
    config_rows = []
    for seed in seeds:
        seed_out = args.out_root / f"seed{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        hmoe_dir = args.hmoe_root / f"seed{seed}" / "target_finetune"
        stage5e_dir = args.stage5e_root / f"seed{seed}"
        support_paths = {
            "hmoe_support": hmoe_dir / "target_support_predictions.csv",
        }
        missing = [str(path) for path in support_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing required final-protocol prediction files:\n" + "\n".join(missing))
        for key, path in support_paths.items():
            hash_rows.append({"seed": seed, "artifact": key, "path": str(path), "sha256": sha256_file(path)})

        hmoe_support = pd.read_csv(support_paths["hmoe_support"])
        stage5e_support_path = stage5e_dir / "target_support_predictions.csv"
        if stage5e_support_path.exists():
            hash_rows.append(
                {
                    "seed": seed,
                    "artifact": "stage5e_support",
                    "path": str(stage5e_support_path),
                    "sha256": sha256_file(stage5e_support_path),
                }
            )
            stage5e_support = pd.read_csv(stage5e_support_path)
        else:
            stage5e_support, rebuilt_hashes = build_stage5e_prediction_from_locked_weights(stage5e_dir, "support")
            for row in rebuilt_hashes:
                hash_rows.append({"seed": seed, **row})
        hmoe_support, stage5e_support = align_pair(hmoe_support, stage5e_support, "support")

        if len(hmoe_support) != 300:
            raise ValueError(f"seed{seed} expected 300 support rows, got {len(hmoe_support)}")
        per_kernel_support = hmoe_support.groupby("kernel").size().to_dict()
        if any(int(value) != 50 for value in per_kernel_support.values()) or len(per_kernel_support) != 6:
            raise ValueError(f"seed{seed} support is not 50/kernel: {per_kernel_support}")

        lambdas, lambda_search = select_lambdas(
            hmoe_support,
            stage5e_support,
            lambda_grid=lambda_grid,
            min_support_rel_gain=args.min_support_rel_gain,
            one_se_tolerance=args.one_se_tolerance,
        )
        stage5i_support = blend_predictions(hmoe_support, stage5e_support, lambdas, seed)
        stage5i_support.to_csv(seed_out / "target_support_predictions.csv", index=False)
        lambda_search.to_csv(seed_out / "stage5i_lambda_search.csv", index=False)

        hmoe_support_metrics = metric_record(hmoe_support, "HMoE_support", args.metric_batch_size)
        stage5e_support_metrics = metric_record(stage5e_support, "Stage5E_support", args.metric_batch_size)
        stage5i_support_metrics = metric_record(stage5i_support, "Stage5I_support", args.metric_batch_size)
        support_summary = pd.DataFrame([hmoe_support_metrics, stage5e_support_metrics, stage5i_support_metrics])
        support_summary.to_csv(seed_out / "stage5i_support_summary.csv", index=False)

        config_rows.append({"seed": seed, **{f"lambda_{target}": lambdas[target] for target in TARGETS}})

        support_hashes = [row for row in hash_rows if row["seed"] == seed]
        lock = {
            "method_id": METHOD_ID,
            "method_name": METHOD_NAME,
            "protocol": "paired_shared_hmoe_kmeans_support",
            "seed": seed,
            "support_rows": int(len(hmoe_support)),
            "support_per_kernel": {str(k): int(v) for k, v in per_kernel_support.items()},
            "anchor": "complete_hmoe_exported_checkpoint_predictions",
            "correction": "Stage5E-DMW-AFRC predictions",
            "formula": "y = y_hmoe + lambda_target * (y_stage5e - y_hmoe)",
            "lambda_grid": lambda_grid,
            "selected_lambdas": lambdas,
            "min_support_rel_gain": args.min_support_rel_gain,
            "one_se_tolerance": args.one_se_tolerance,
            "query_labels_used_for_selection": False,
            "query_predictions_read_before_selection": False,
            "support_input_hashes": support_hashes,
            "scientific_boundary": (
                "HMoE-augmented diagnostic, not PureNew5. It uses complete HMoE "
                "as an inference anchor and must be reported separately."
            ),
        }
        write_json(seed_out / "selection_lock.json", lock)

        query_paths = {
            "hmoe_query": hmoe_dir / "target_query_predictions.csv",
            "stage5e_query": stage5e_dir / "target_query_predictions.csv",
        }
        missing = [str(path) for path in query_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing required final-protocol query prediction files:\n" + "\n".join(missing))
        for key, path in query_paths.items():
            hash_rows.append({"seed": seed, "artifact": key, "path": str(path), "sha256": sha256_file(path)})
        hmoe_query = pd.read_csv(query_paths["hmoe_query"])
        stage5e_query = pd.read_csv(query_paths["stage5e_query"])
        hmoe_query, stage5e_query = align_pair(hmoe_query, stage5e_query, "query")
        if len(hmoe_query) != 837:
            raise ValueError(f"seed{seed} expected 837 query rows, got {len(hmoe_query)}")

        stage5i_query = blend_predictions(hmoe_query, stage5e_query, lambdas, seed)
        stage5i_query.to_csv(seed_out / "target_query_predictions.csv", index=False)

        hmoe_query_metrics = metric_record(hmoe_query, "HMoE", args.metric_batch_size)
        stage5e_query_metrics = metric_record(stage5e_query, "Stage5E", args.metric_batch_size)
        stage5i_query_metrics = metric_record(stage5i_query, METHOD_NAME, args.metric_batch_size)
        query_summary = pd.DataFrame([hmoe_query_metrics, stage5e_query_metrics, stage5i_query_metrics])
        query_summary.to_csv(seed_out / "stage5i_query_summary.csv", index=False)
        hmoe_report_from_predictions(stage5i_query).to_csv(seed_out / "target_query_hmoe_report.csv", index=False)

        hmoe_loss = float(hmoe_query_metrics["hmoe_test_loss"])
        per_seed_rows.append(method_row(seed, "HMoE", hmoe_query_metrics, hmoe_loss))
        per_seed_rows.append(method_row(seed, "Stage5E", stage5e_query_metrics, hmoe_loss))
        per_seed_rows.append(method_row(seed, METHOD_NAME, stage5i_query_metrics, hmoe_loss))

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(args.out_root / "stage5i_per_seed.csv", index=False)
    pd.DataFrame(config_rows).to_csv(args.out_root / "stage5i_selected_lambdas.csv", index=False)
    summary_rows = []
    for method, group in per_seed.groupby("method", sort=False):
        row = {"method": method, "n_seeds": int(len(group))}
        for metric in [
            "hmoe_test_loss",
            "perf_rmse",
            "util_LUT_rmse",
            "util_FF_rmse",
            "util_DSP_rmse",
            "util_BRAM_rmse",
            "overall_rmse",
            "hmoe_report_rmse_sum",
            "delta_vs_hmoe",
            "relative_change_vs_hmoe",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        row["wins_vs_hmoe"] = int(group["better_than_hmoe"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_root / "stage5i_mean_std.csv", index=False)

    all_hash_rows = list(hash_rows)
    for path in sorted(args.out_root.glob("**/*")):
        if path.is_file() and path.name != "RESULT_FILE_SHA256.csv":
            all_hash_rows.append({"seed": "", "artifact": path.name, "path": str(path), "sha256": sha256_file(path)})
    pd.DataFrame(all_hash_rows).to_csv(args.out_root / "RESULT_FILE_SHA256.csv", index=False)

    readme = f"""# {METHOD_NAME}

Method ID: `{METHOD_ID}`

Protocol: `paired_shared_hmoe_kmeans_support`.

This is an HMoE-augmented diagnostic, not the PureNew5 primary method. It uses
complete HMoE as the inference anchor and applies a strongly constrained
target-specific correction toward Stage5E:

```text
y = y_hmoe + lambda_target * (y_stage5e - y_hmoe)
```

`lambda_target` is selected using support labels only from the fixed grid
`{lambda_grid}`. Query predictions and labels are not read before each seed's
`selection_lock.json` is written.
"""
    (args.out_root / "README.md").write_text(readme, encoding="utf-8")

    if args.archive_root is not None:
        write_archive(args.out_root, args.archive_root)

    print("Stage5I per-seed:")
    print(per_seed[["seed", "method", "hmoe_test_loss", "delta_vs_hmoe", "better_than_hmoe"]].to_string(index=False))
    print("\nStage5I mean/std:")
    print(summary[["method", "hmoe_test_loss_mean", "hmoe_test_loss_std", "overall_rmse_mean", "wins_vs_hmoe"]].to_string(index=False))
    print(f"\nWrote: {args.out_root}")
    if args.archive_root is not None:
        print(f"Archived summaries: {args.archive_root}")


if __name__ == "__main__":
    main()
