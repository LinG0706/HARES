from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from hmoe_metrics import hmoe_summary_columns, hmoe_test_loss_from_predictions
from release_metrics import compute_metrics
from release_core import (
    TARGETS, load_candidates, model_names_from_frames, parse_candidate,
    parse_design_key, parse_float_grid, parse_int_grid, prediction_matrix,
    predict_with_models, split_support_indices, true_values, write_json,
    write_weights,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_hashes(candidate_specs: list[str], splits: tuple[str, ...]) -> list[dict]:
    rows = []
    for spec in candidate_specs:
        name, path = parse_candidate(spec)
        for split in splits:
            csv_path = Path(path) / f"target_{split}_predictions.csv"
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "path": str(csv_path),
                    "sha256": sha256_file(csv_path),
                }
            )
    return rows


def verify_query_artifacts(
    candidate_specs: list[str],
) -> list[dict]:
    rows = []
    for spec in candidate_specs:
        name, path = parse_candidate(spec)
        query_path = Path(path) / "target_query_predictions.csv"
        current_hash = sha256_file(query_path)
        row = {"model": name, "path": str(query_path), "sha256": current_hash}
        rows.append(row)
    return rows


def read_query_metadata(first_candidate_spec: str) -> pd.DataFrame:
    _, path = parse_candidate(first_candidate_spec)
    csv_path = Path(path) / "target_query_predictions.csv"
    metadata = pd.read_csv(csv_path, usecols=["sample_id", "kernel", "key"])
    if metadata["sample_id"].duplicated().any():
        raise ValueError(f"duplicate query sample_id in {csv_path}")
    return metadata.reset_index(drop=True)


def signed_log2(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log2(1.0 + np.abs(values))


def design_frames(support_base: pd.DataFrame, query_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["sample_id", "kernel", "key"]
    for name, frame in [("support", support_base), ("query", query_base)]:
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"{name} predictions lack design metadata: {missing}")
        if frame["key"].isna().any():
            raise ValueError(f"{name} contains missing design keys")

    all_keys = support_base["key"].tolist() + query_base["key"].tolist()
    parsed = pd.DataFrame([parse_design_key(key) for key in all_keys]).fillna(0.0)
    if parsed.empty:
        raise ValueError("no design features could be parsed from prediction keys")
    numeric_cols = [col for col in parsed.columns if col.endswith("::num")]
    if numeric_cols:
        parsed.loc[:, numeric_cols] = signed_log2(parsed[numeric_cols].to_numpy(dtype=np.float64))
    parsed = parsed.astype(np.float64)
    n_support = len(support_base)
    return parsed.iloc[:n_support].reset_index(drop=True), parsed.iloc[n_support:].reset_index(drop=True)


def support_design_mass(support_base: pd.DataFrame, query_base: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict]:
    support_design, query_design = design_frames(support_base, query_base)
    masses = np.zeros(len(support_base), dtype=np.float64)
    assignment_rows = []
    feature_count = support_design.shape[1]

    for kernel, support_group in support_base.groupby("kernel", sort=True):
        support_idx = support_group.index.to_numpy(dtype=int)
        query_idx = query_base.index[query_base["kernel"] == kernel].to_numpy(dtype=int)
        if len(support_idx) == 0 or len(query_idx) == 0:
            raise ValueError(f"empty support/query design group for kernel={kernel}")

        support_values = support_design.iloc[support_idx].to_numpy(dtype=np.float64)
        query_values = query_design.iloc[query_idx].to_numpy(dtype=np.float64)
        median = np.median(support_values, axis=0)
        q25 = np.percentile(support_values, 25, axis=0)
        q75 = np.percentile(support_values, 75, axis=0)
        scale = q75 - q25
        std = np.std(support_values, axis=0)
        scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
        support_scaled = (support_values - median) / scale
        query_scaled = (query_values - median) / scale

        distances = cdist(query_scaled, support_scaled, metric="euclidean")
        assigned = np.zeros(len(support_idx), dtype=np.float64)
        for row in distances:
            minimum = float(np.min(row))
            tolerance = 1e-12 + 1e-9 * max(1.0, abs(minimum))
            ties = np.flatnonzero(row <= minimum + tolerance)
            assigned[ties] += 1.0 / len(ties)

        raw_mass = np.sqrt(1.0 + assigned)
        normalized = raw_mass / np.mean(raw_mass)
        masses[support_idx] = normalized
        for local_idx, global_idx in enumerate(support_idx):
            assignment_rows.append(
                {
                    "sample_id": support_base.loc[global_idx, "sample_id"],
                    "kernel": kernel,
                    "assigned_query_mass": float(assigned[local_idx]),
                    "raw_mass": float(raw_mass[local_idx]),
                    "normalized_mass": float(normalized[local_idx]),
                }
            )

    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("invalid support design masses")
    stats = {
        "feature_count": int(feature_count),
        "mass_min": float(np.min(masses)),
        "mass_max": float(np.max(masses)),
        "mass_mean": float(np.mean(masses)),
        "assigned_query_total": float(sum(row["assigned_query_mass"] for row in assignment_rows)),
    }
    return masses, pd.DataFrame(assignment_rows), stats


def fit_weighted_convex(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, alpha: float, prior: np.ndarray):
    n_models = X.shape[1]
    weights_obs = np.asarray(sample_weights, dtype=np.float64)
    weights_obs = weights_obs / np.sum(weights_obs)
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / np.sum(prior)

    def objective(weights):
        residual = X @ weights - y
        delta = weights - prior
        return float(np.sum(weights_obs * residual * residual) + alpha * (delta @ delta))

    def gradient(weights):
        residual = X @ weights - y
        return 2.0 * (X.T @ (weights_obs * residual) + alpha * (weights - prior))

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones(n_models, dtype=np.float64),
        }
    ]
    result = minimize(
        objective,
        prior.copy(),
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, 1.0) for _ in range(n_models)],
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )
    candidate = np.clip(result.x, 0.0, 1.0)
    if not np.all(np.isfinite(candidate)) or candidate.sum() <= 0:
        candidate = prior.copy()
    return candidate / candidate.sum(), 0.0


def empty_models(scope: str, frames):
    base = frames[0]
    if scope == "global":
        return {("global", target): None for target in TARGETS}
    if scope == "per_kernel":
        return {(kernel, target): None for kernel in sorted(base["kernel"].unique()) for target in TARGETS}
    raise ValueError(f"unsupported scope: {scope}")


def fit_models(frames, indices, design_masses, scope: str, alpha: float, masks=None):
    model_names = model_names_from_frames(frames)
    n_models = len(model_names)
    base = frames[0]
    models = empty_models(scope, frames)

    def fit_one(key, target, local_indices):
        mask = np.ones(n_models, dtype=bool) if masks is None else np.asarray(masks[key], dtype=bool).copy()
        if not np.any(mask):
            raise ValueError(f"empty Stage5E mask for {key}")
        allowed = np.flatnonzero(mask)
        output_weights = np.zeros(n_models, dtype=np.float64)
        if len(allowed) == 1:
            output_weights[allowed[0]] = 1.0
            return {"weights": output_weights, "bias": 0.0}

        X = prediction_matrix(frames, target, local_indices)[:, allowed]
        y = true_values(base, target, local_indices)
        obs_weights = design_masses[local_indices]
        prior = np.full(len(allowed), 1.0 / len(allowed), dtype=np.float64)
        local_weights, bias = fit_weighted_convex(X, y, obs_weights, alpha, prior)
        output_weights[allowed] = local_weights
        return {"weights": output_weights, "bias": bias}

    if scope == "global":
        for target in TARGETS:
            key = ("global", target)
            models[key] = fit_one(key, target, indices)
        return models

    train_base = base.iloc[indices]
    for kernel, group in train_base.groupby("kernel", sort=True):
        local_indices = group.index.to_numpy(dtype=int)
        for target in TARGETS:
            key = (kernel, target)
            models[key] = fit_one(key, target, local_indices)
    return models


def models_to_cube(models, keys, n_models):
    return np.vstack([np.asarray(models[key]["weights"], dtype=np.float64) for key in keys]).reshape(len(keys), n_models)


def build_anchorfree_masks(weight_cubes, keys, n_models, selection_threshold, weight_floor):
    stack = np.stack(weight_cubes, axis=0)
    mean_weight = np.mean(stack, axis=0)
    std_weight = np.std(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean_weight)
    selection_rate = np.mean(stack >= weight_floor, axis=0)
    masks = {}
    for key_idx, key in enumerate(keys):
        allowed = (selection_rate[key_idx] >= selection_threshold) & (mean_weight[key_idx] >= weight_floor)
        if not np.any(allowed):
            allowed[int(np.argmax(mean_weight[key_idx]))] = True
        masks[key] = allowed
    return masks, mean_weight, std_weight, selection_rate


def weighted_target_loss(predictions: pd.DataFrame, sample_weights: np.ndarray) -> float:
    weights = np.asarray(sample_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    total = 0.0
    for target in TARGETS:
        residual = (
            predictions[f"y_pred_{target}"].to_numpy(dtype=np.float64)
            - predictions[f"y_true_{target}"].to_numpy(dtype=np.float64)
        )
        total += float(np.sum(weights * residual * residual))
    return total


def allowed_stats(masks):
    counts = np.asarray([int(np.sum(mask)) for mask in masks.values()], dtype=np.float64)
    return {
        "mean_allowed": float(np.mean(counts)),
        "min_allowed": int(np.min(counts)),
        "max_allowed": int(np.max(counts)),
    }


def evaluate_config(frames, design_masses, fold_items, keys, n_models, scope, alpha, threshold, floor, batch_size):
    weighted_losses = []
    unweighted_losses = []
    allowed_counts = []
    fold_rows = []
    for fold_idx, item in enumerate(fold_items):
        other_cubes = [fold["weight_cube"] for idx, fold in enumerate(fold_items) if idx != fold_idx]
        masks, _, _, _ = build_anchorfree_masks(other_cubes, keys, n_models, threshold, floor)
        models = fit_models(frames, item["train_idx"], design_masses, scope, alpha, masks)
        val_frames = [frame.iloc[item["val_idx"]].reset_index(drop=True) for frame in frames]
        val_pred = predict_with_models(val_frames, models, scope, "stage5e_support_cv")
        weighted_loss = weighted_target_loss(val_pred, design_masses[item["val_idx"]])
        metric_row, _ = compute_metrics(val_pred, f"stage5e_fold{fold_idx}", batch_size)
        stats = allowed_stats(masks)
        weighted_losses.append(weighted_loss)
        unweighted_losses.append(float(metric_row["hmoe_test_loss"]))
        allowed_counts.append(stats["mean_allowed"])
        fold_rows.append(
            {
                "fold_idx": fold_idx,
                "seed": item["seed"],
                "scope": scope,
                "alpha": alpha,
                "selection_threshold": threshold,
                "weight_floor": floor,
                "weighted_val_loss": weighted_loss,
                "unweighted_val_hmoe_loss": float(metric_row["hmoe_test_loss"]),
                **stats,
            }
        )

    mean = float(np.mean(weighted_losses))
    std = float(np.std(weighted_losses, ddof=1)) if len(weighted_losses) > 1 else 0.0
    return {
        "scope": scope,
        "alpha": alpha,
        "selection_threshold": threshold,
        "weight_floor": floor,
        "weighted_cv_mean": mean,
        "weighted_cv_std": std,
        "weighted_cv_max": float(np.max(weighted_losses)),
        "unweighted_cv_mean": float(np.mean(unweighted_losses)),
        "mean_allowed": float(np.mean(allowed_counts)),
        "robust_score": mean + 0.5 * std,
        "fold_rows": fold_rows,
    }


def mask_table(keys, model_names, masks, mean_weight, std_weight, selection_rate):
    rows = []
    for key_idx, key in enumerate(keys):
        for model_idx, model in enumerate(model_names):
            rows.append(
                {
                    "scope_key": key[0],
                    "target": key[1],
                    "model": model,
                    "allowed": bool(masks[key][model_idx]),
                    "mean_weight": float(mean_weight[key_idx, model_idx]),
                    "std_weight": float(std_weight[key_idx, model_idx]),
                    "selection_rate": float(selection_rate[key_idx, model_idx]),
                }
            )
    return pd.DataFrame(rows)


def select_one_se(config_df: pd.DataFrame, n_folds: int) -> dict:
    ranked = config_df.sort_values(["robust_score", "weighted_cv_mean", "weighted_cv_std"]).reset_index(drop=True)
    best = ranked.iloc[0]
    threshold = float(best["robust_score"] + best["weighted_cv_std"] / math.sqrt(n_folds))
    eligible = ranked[ranked["robust_score"] <= threshold].copy()
    eligible["scope_rank"] = eligible["scope"].map({"global": 0, "per_kernel": 1})
    selected = eligible.sort_values(
        ["scope_rank", "alpha", "selection_threshold", "weight_floor", "robust_score"],
        ascending=[True, False, False, False, True],
    ).iloc[0]
    output = selected.drop(labels=["scope_rank"]).to_dict()
    output["best_robust_score"] = float(best["robust_score"])
    output["one_se_threshold"] = threshold
    output["n_one_se_eligible"] = int(len(eligible))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scopes", nargs="+", choices=["global", "per_kernel"], default=["global", "per_kernel"])
    parser.add_argument("--alpha-grid", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--cv-seeds", default="1-50")
    parser.add_argument("--support-val-per-kernel", type=int, default=10)
    parser.add_argument("--selection-threshold-grid", default="0.70,0.85,0.95")
    parser.add_argument("--weight-floor-grid", default="0.03,0.05,0.10")
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--method-id", default="FSS-R2-M1")
    parser.add_argument("--method-name", default="Stage5E-DMW-AFRC")
    parser.add_argument("--artifact-prefix", default="stage5e")
    parser.add_argument("--prediction-model", default="stage5e_dmw_afrc")
    parser.add_argument("--outer-seed", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    support_frames = load_candidates(args.candidates, "support")
    model_names = model_names_from_frames(support_frames)

    support_base = support_frames[0].reset_index(drop=True)
    query_meta = read_query_metadata(args.candidates[0])
    design_masses, mass_table, mass_stats = support_design_mass(support_base, query_meta)
    mass_table.to_csv(args.out_dir / f"{args.artifact_prefix}_design_mass.csv", index=False)

    alphas = parse_float_grid(args.alpha_grid)
    cv_seeds = parse_int_grid(args.cv_seeds)
    thresholds = parse_float_grid(args.selection_threshold_grid)
    floors = parse_float_grid(args.weight_floor_grid)
    n_models = len(model_names)
    config_rows = []
    fold_rows = []
    fold_cache = {}

    for scope in args.scopes:
        keys = list(empty_models(scope, support_frames).keys())
        for alpha in alphas:
            fold_items = []
            for seed in cv_seeds:
                train_idx, val_idx = split_support_indices(support_base, args.support_val_per_kernel, seed)
                unmasked = fit_models(support_frames, train_idx, design_masses, scope, alpha, masks=None)
                fold_items.append(
                    {
                        "seed": seed,
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                        "weight_cube": models_to_cube(unmasked, keys, n_models),
                    }
                )
            fold_cache[(scope, alpha)] = (keys, fold_items)
            for threshold in thresholds:
                for floor in floors:
                    result = evaluate_config(
                        support_frames,
                        design_masses,
                        fold_items,
                        keys,
                        n_models,
                        scope,
                        alpha,
                        threshold,
                        floor,
                        args.metric_batch_size,
                    )
                    fold_rows.extend(result.pop("fold_rows"))
                    config_rows.append(result)

    config_df = pd.DataFrame(config_rows).sort_values(["robust_score", "weighted_cv_mean", "weighted_cv_std"])
    config_df.to_csv(args.out_dir / f"{args.artifact_prefix}_config_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / f"{args.artifact_prefix}_fold_summary.csv", index=False)
    selected = select_one_se(config_df, len(cv_seeds))
    selected_scope = str(selected["scope"])
    selected_alpha = float(selected["alpha"])
    keys, fold_items = fold_cache[(selected_scope, selected_alpha)]
    cubes = [item["weight_cube"] for item in fold_items]
    masks, mean_weight, std_weight, selection_rate = build_anchorfree_masks(
        cubes,
        keys,
        n_models,
        float(selected["selection_threshold"]),
        float(selected["weight_floor"]),
    )
    mask_table(keys, model_names, masks, mean_weight, std_weight, selection_rate).to_csv(
        args.out_dir / f"{args.artifact_prefix}_stability_mask.csv", index=False
    )
    full_idx = support_base.index.to_numpy(dtype=int)
    final_models = fit_models(support_frames, full_idx, design_masses, selected_scope, selected_alpha, masks)
    write_weights(final_models, model_names, args.out_dir / f"{args.artifact_prefix}_convex_weights.csv")

    lock = {
        "method_id": args.method_id,
        "method": args.method_name,
        "selected_config": selected,
        "allowed_stats": allowed_stats(masks),
        "design_mass": mass_stats,
        "query_features_used_for_selection": ["sample_id", "kernel", "key"],
        "query_predictions_used_for_selection": False,
        "query_labels_used_for_selection": False,
        "outer_seed": args.outer_seed,
        "support_candidate_hashes": candidate_hashes(args.candidates, ("support",)),
        "cv_seeds": cv_seeds,
        "support_val_per_kernel": args.support_val_per_kernel,
        "alpha_grid": alphas,
        "selection_threshold_grid": thresholds,
        "weight_floor_grid": floors,
    }
    lock_path = args.out_dir / "selection_lock.json"
    write_json(lock_path, lock)

    query_candidate_hashes = verify_query_artifacts(args.candidates)
    query_frames = load_candidates(args.candidates, "query")
    if model_names != model_names_from_frames(query_frames):
        raise ValueError("support/query candidate order differs")
    if query_frames[0]["sample_id"].astype(str).tolist() != query_meta["sample_id"].astype(str).tolist():
        raise ValueError("locked query metadata order differs from aligned query predictions")

    query_pred = predict_with_models(query_frames, final_models, selected_scope, args.prediction_model)
    query_pred.to_csv(args.out_dir / "target_query_predictions.csv", index=False)

    summary_rows = []
    row, _ = compute_metrics(query_pred, f"target_query_{args.artifact_prefix}", args.metric_batch_size)
    summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / f"{args.artifact_prefix}_query_summary.csv", index=False)

    metrics = {
        **lock,
        "query_candidate_hashes": query_candidate_hashes,
        "query_size": int(len(query_pred)),
        "target_query_hmoe": hmoe_test_loss_from_predictions(query_pred, batch_size=args.metric_batch_size),
        "target_query_hmoe_summary": hmoe_summary_columns(query_pred),
    }
    write_json(args.out_dir / f"{args.artifact_prefix}_metrics.json", metrics)

    print("Selection lock written before query evaluation:", args.out_dir / "selection_lock.json")
    print("Selected config:", selected)
    print("Design mass:", mass_stats)
    print("\nQuery summary")
    print(
        summary[["method", "hmoe_test_loss", "perf_rmse", "overall_rmse", "hmoe_report_rmse_sum"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"Saved {args.method_name} outputs to:", args.out_dir)


if __name__ == "__main__":
    main()
