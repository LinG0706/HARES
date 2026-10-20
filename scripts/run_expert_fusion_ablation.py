"""Reproduce the paper's uniform/learned pool x anchored-shrinkage ablation.

The learned pool replays the released support-fitted weights and expert masks.
All requested seeds' shrinkage selections are locked before query evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_protocol import TARGETS, check_prediction, manifest_split
from release_core import prediction_matrix, predict_with_models, write_json
from summarize_paper_results import mse_row
from train_stage5i_hmoe_shrinkage import (
    blend_predictions, select_lambdas, stage5e_models_from_weights,
)


EXPERTS = ["harp", "graphgps", "exphormer", "polynormer", "gatedgcn"]
VARIANTS = {
    "anchor_only": ("HMoE only", "--"),
    "uniform_direct": ("Uniform pool", "No"),
    "learned_direct": ("Learned pool", "No"),
    "uniform_anchored": ("Uniform pool", "Yes"),
    "learned_anchored": ("Learned pool (HARES)", "Yes"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uniform_pool(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) != len(EXPERTS):
        raise ValueError("The uniform pool requires all five correction experts")
    result = frames[0].copy()
    for target in TARGETS:
        result[f"y_pred_{target}"] = prediction_matrix(frames, target) @ np.full(5, 0.2)
    result["model"] = "Uniform pool"
    return result


def load_inputs(root: Path, seed: int, split: str) -> tuple:
    reference = manifest_split(root, seed, split)
    pool_dir = root / f"artifacts/inputs/stage5e/seed{seed}"
    lock_path = pool_dir / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("query_labels_used_for_selection") is not False:
        raise ValueError("Learned-pool weights must be selected using support labels only")
    entries = [item for item in lock["candidate_hashes"] if item["split"] == split]
    if [item["model"] for item in entries] != EXPERTS:
        raise ValueError("Frozen correction expert names/order differ from the paper")
    files = [lock_path, pool_dir / "stage5e_convex_weights.csv"]
    frames = []
    for item in entries:
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Candidate hash mismatch: {path}")
        frames.append(check_prediction(path, reference, seed))
        files.append(path)
    anchor_path = root / f"artifacts/inputs/hmoe/seed{seed}/target_finetune/target_{split}_predictions.csv"
    anchor = check_prediction(anchor_path, reference, seed)
    files.append(anchor_path)
    if split == "support":
        models, scope = stage5e_models_from_weights(pool_dir, EXPERTS)
        learned = predict_with_models(frames, models, scope, "Learned pool")
    else:
        # Preserve the original query exports exactly, as in the main HARES replay.
        path = pool_dir / "target_query_predictions.csv"
        learned = check_prediction(path, reference, seed)
        files.append(path)
    uniform = uniform_pool(frames)
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    return anchor, uniform, learned, hashes


def write_table(summary: pd.DataFrame, path: Path, n_seeds: int) -> None:
    count = "five" if n_seeds == 5 else str(n_seeds)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Ablation of expert fusion and HMoE-anchored shrinkage. "
        + r"Total MSE is reported as mean $\pm$ sample standard deviation over " + count + " seeds.}",
        r"\label{tab:ablation}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.06}", r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule", r"\textbf{Expert fusion} & \textbf{Shrinkage} & \textbf{Total MSE}\\",
        r"\midrule",
    ]
    for row in summary.to_dict(orient="records"):
        label = row["method"]
        mean, std = row["total_mse_mean"], row["total_mse_std"]
        value = f"{mean:.4f}\\pm{std:.4f}"
        if row["variant"] == "learned_anchored":
            label = r"\textbf{" + label + "}"
            value = r"\mathbf{" + f"{mean:.4f}" + r"}\pm" + f"{std:.4f}"
        lines.append(label + " & " + row["shrinkage"] + " & $" + value + r"$\\")
        if row["variant"] == "anchor_only":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{-3mm}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--seeds", default="1 2 3 4 5")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output_dir).resolve()
    inputs = root / "artifacts/inputs"
    if output.is_relative_to(inputs) or inputs.is_relative_to(output):
        raise ValueError("Output directory must not overlap frozen prediction inputs")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= {1, 2, 3, 4, 5}:
        raise ValueError("Use unique seeds from the released five-seed protocol")
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for sample standard deviations")
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs/final_protocol.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["shrinkage"]
    definition = {
        "factors": ["uniform versus learned expert fusion", "without versus with HMoE-anchored shrinkage"],
        "reference": "HMoE only",
        "seeds": seeds,
        "uniform_weights": dict.fromkeys(EXPERTS, 0.2),
        "learned_pool": "replay frozen support-fitted weights and expert masks; no neural retraining",
        "shrinkage": config,
        "joint_factor_note": "Anchoring and shrinkage are tested jointly, not as isolated components.",
        "all_four_cells_reported": True,
        "query_labels_used_for_selection": False,
        "post_hoc_diagnostic": True,
        "script_sha256": sha256(Path(__file__)),
        "protocol_config_sha256": sha256(config_path),
    }
    definition_path = output / "definition.json"
    write_json(definition_path, definition)
    selections, lock_hashes, all_hashes = {}, {}, {}
    for seed in seeds:
        anchor, uniform, learned, hashes = load_inputs(root, seed, "support")
        folder = output / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        lambdas = {}
        selected_artifacts = {}
        for name, correction in (("uniform", uniform), ("learned", learned)):
            values, search = select_lambdas(
                anchor, correction, config["lambda_grid"],
                config["min_support_relative_gain"], config["one_se_tolerance"],
            )
            lambdas[name] = values
            correction.to_csv(folder / f"{name}_support_predictions.csv", index=False)
            search.to_csv(folder / f"{name}_lambda_search.csv", index=False)
            for suffix in ("support_predictions", "lambda_search"):
                name_out = f"{name}_{suffix}.csv"
                selected_artifacts[name_out] = sha256(folder / name_out)
        lock = {
            "seed": seed, "definition_sha256": sha256(definition_path),
            "selected_lambdas": lambdas, "support_input_hashes": hashes,
            "selected_artifact_hashes": selected_artifacts,
            "query_labels_used_for_selection": False,
            "query_predictions_read_before_selection": False,
        }
        write_json(folder / "selection_lock.json", lock)
        selections[seed] = lock
        lock_hashes[str(seed)] = sha256(folder / "selection_lock.json")
        all_hashes.update(hashes)
        print(f"seed{seed}: support selections locked", flush=True)
    marker = {"definition_sha256": sha256(definition_path), "selection_lock_hashes": lock_hashes}
    marker_path = output / "all_support_selections_locked.json"
    write_json(marker_path, marker)

    records = []
    for seed in seeds:
        folder, lock = output / f"seed{seed}", selections[seed]
        if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise ValueError("All-seed selection marker changed before query evaluation")
        if sha256(folder / "selection_lock.json") != lock_hashes[str(seed)]:
            raise ValueError("Support selection lock changed")
        for name, expected in lock["selected_artifact_hashes"].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"Selected artifact changed: {folder / name}")
        for relative, expected in lock["support_input_hashes"].items():
            if sha256(root / relative) != expected:
                raise ValueError(f"Support input changed after selection: {relative}")
        anchor, uniform, learned, hashes = load_inputs(root, seed, "query")
        for relative in hashes.keys() & all_hashes.keys():
            if hashes[relative] != all_hashes[relative]:
                raise ValueError(f"Frozen input changed after selection: {relative}")
        all_hashes.update(hashes)
        frames = {"anchor_only": anchor, "uniform_direct": uniform, "learned_direct": learned}
        for name, correction in (("uniform", uniform), ("learned", learned)):
            frames[f"{name}_anchored"] = blend_predictions(anchor, correction, lock["selected_lambdas"][name], seed)
        for variant, (method, shrinkage) in VARIANTS.items():
            frame = frames[variant].copy()
            frame["model"], frame["seed"] = method, seed
            # Drop derived errors and identifiers inherited from a different variant.
            frame = frame[["sample_id", "kernel", "seed", "model"] +
                          [f"y_{kind}_{t}" for t in TARGETS for kind in ("true", "pred")]]
            destination = folder / variant
            destination.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination / "target_query_predictions.csv", index=False)
            records.append({**mse_row(frame, method, seed), "variant": variant, "shrinkage": shrinkage})
        print(f"seed{seed}: five query contrasts evaluated", flush=True)
    for relative, expected in all_hashes.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen input changed during evaluation: {relative}")
    per_seed = pd.DataFrame(records)
    summary = []
    for variant, (method, shrinkage) in VARIANTS.items():
        group = per_seed.loc[per_seed["variant"] == variant]
        row = {"variant": variant, "method": method, "shrinkage": shrinkage, "n_seeds": len(group)}
        for metric in [f"{t}_mse" for t in TARGETS] + ["total_mse"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary.append(row)
    summary = pd.DataFrame(summary)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    summary.to_csv(output / "mean_std.csv", index=False)
    write_table(summary, output / "ablation_table.tex", len(seeds))
    pd.DataFrame([{"path": p, "sha256": h} for p, h in sorted(all_hashes.items())]).to_csv(
        output / "input_checksums.csv", index=False)
    write_json(output / "completion.json", {
        "complete": True, "all_four_cells_retained": True, "reference": "HMoE only",
        "seeds": seeds, "query_rows_per_seed": 837, "query_contrasts": len(records),
        "all_support_selections_locked_before_query_evaluation": True,
        "query_labels_used_for_selection": False, "frozen_inputs_unchanged": True,
        "definition_sha256": sha256(definition_path),
    })
    print(summary[["method", "shrinkage", "total_mse_mean", "total_mse_std"]].to_string(index=False))
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
