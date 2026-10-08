from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from hmoe_metrics import hmoe_report_from_predictions


TARGET_METHOD = "Stage5I-HMoE-target-shrinkage"


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.replace(",", " ").split() if item.strip()]


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


def read_candidate_catalog(stage5e_seed_dir: Path) -> dict[str, Path]:
    lock_path = stage5e_seed_dir / "selection_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    catalog: dict[str, Path] = {}
    for item in lock.get("candidate_hashes", []):
        if item.get("split") != "support":
            continue
        model = str(item["model"])
        path = resolve_recorded_path(str(item["path"])).parent
        support = path / "target_support_predictions.csv"
        query = path / "target_query_predictions.csv"
        if not support.exists() or not query.exists():
            raise FileNotFoundError(f"missing candidate predictions for {model}: {support} / {query}")
        catalog[model] = path
    required = {"graphgps", "exphormer", "polynormer", "gatedgcn"}
    missing = sorted(required - set(catalog))
    if missing:
        raise ValueError(f"{stage5e_seed_dir} lacks required new-GNN candidates: {missing}")
    return catalog


def pool_definitions() -> dict[str, list[str]]:
    return {
        "full_harp_new4": [
            "harp_hmoe_support_paired",
            "graphgps",
            "exphormer",
            "polynormer",
            "gatedgcn",
        ],
        "loo_no_graphgps": [
            "harp_hmoe_support_paired",
            "exphormer",
            "polynormer",
            "gatedgcn",
        ],
        "loo_no_exphormer": [
            "harp_hmoe_support_paired",
            "graphgps",
            "polynormer",
            "gatedgcn",
        ],
        "loo_no_polynormer": [
            "harp_hmoe_support_paired",
            "graphgps",
            "exphormer",
            "gatedgcn",
        ],
        "loo_no_gatedgcn": [
            "harp_hmoe_support_paired",
            "graphgps",
            "exphormer",
            "polynormer",
        ],
        "loo_no_harp": [
            "graphgps",
            "exphormer",
            "polynormer",
            "gatedgcn",
        ],
        "harp_only_correction": ["harp_hmoe_support_paired"],
    }


def candidate_specs(catalog: dict[str, Path], names: list[str]) -> list[str]:
    specs = []
    for name in names:
        if name not in catalog:
            raise KeyError(f"candidate {name} not found in catalog: {sorted(catalog)}")
        specs.append(f"{name}={catalog[name]}")
    return specs


def stage5e_args_from_lock(stage5e_seed_dir: Path) -> dict[str, str]:
    lock = json.loads((stage5e_seed_dir / "selection_lock.json").read_text(encoding="utf-8"))
    scopes = " ".join(["global", "per_kernel"])
    if "selected_config" in lock and lock["selected_config"].get("scope") == "per_kernel":
        scopes = "global per_kernel"
    cv_seeds = [int(x) for x in lock.get("cv_seeds", list(range(1, 51)))]
    if cv_seeds == list(range(min(cv_seeds), max(cv_seeds) + 1)):
        cv_seed_text = f"{min(cv_seeds)}-{max(cv_seeds)}"
    else:
        cv_seed_text = ",".join(str(x) for x in cv_seeds)
    return {
        "scopes": scopes,
        "alpha_grid": ",".join(str(x) for x in lock.get("alpha_grid", [0.0001, 0.001, 0.01, 0.1, 1.0])),
        "cv_seeds": cv_seed_text,
        "support_val_per_kernel": str(lock.get("support_val_per_kernel", 10)),
        "selection_threshold_grid": ",".join(str(x) for x in lock.get("selection_threshold_grid", [0.70, 0.85, 0.95])),
        "weight_floor_grid": ",".join(str(x) for x in lock.get("weight_floor_grid", [0.03, 0.05, 0.10])),
    }


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("\n$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def collect_summary(out_root: Path, pools: dict[str, list[str]], seeds: list[int]) -> None:
    rows = []
    lambda_rows = []
    for pool_name, experts in pools.items():
        summary_path = out_root / pool_name / "stage5i" / "stage5i_mean_std.csv"
        per_seed_path = out_root / pool_name / "stage5i" / "stage5i_per_seed.csv"
        lambda_path = out_root / pool_name / "stage5i" / "stage5i_selected_lambdas.csv"
        if not summary_path.exists() or not per_seed_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        selected = summary[summary["method"] == TARGET_METHOD].copy()
        if selected.empty:
            continue
        selected.insert(0, "pool", pool_name)
        selected.insert(1, "experts", "+".join(experts))
        selected.insert(2, "n_experts", len(experts))
        rows.append(selected)
        per_seed = pd.read_csv(per_seed_path)
        per_seed = per_seed[per_seed["method"] == TARGET_METHOD].copy()
        mse_values = []
        for seed in seeds:
            prediction_path = out_root / pool_name / "stage5i" / f"seed{seed}" / "target_query_predictions.csv"
            if not prediction_path.exists():
                continue
            prediction = pd.read_csv(prediction_path)
            report = hmoe_report_from_predictions(prediction)
            total_mse = float(report.loc[report["target"] == "tot/avg", "mse"].iloc[0])
            mse_values.append({"seed": seed, "hmoe_report_mse_sum": total_mse})
        if mse_values:
            mse_frame = pd.DataFrame(mse_values)
            per_seed = per_seed.merge(mse_frame, on="seed", how="left", validate="one_to_one")
            selected["hmoe_report_mse_sum_mean"] = float(mse_frame["hmoe_report_mse_sum"].mean())
            selected["hmoe_report_mse_sum_std"] = (
                float(mse_frame["hmoe_report_mse_sum"].std(ddof=1))
                if len(mse_frame) > 1
                else 0.0
            )
        per_seed.insert(0, "pool", pool_name)
        per_seed.insert(1, "experts", "+".join(experts))
        per_seed.to_csv(out_root / pool_name / "stage5i_ablation_per_seed_only.csv", index=False)
        if lambda_path.exists():
            lambdas = pd.read_csv(lambda_path)
            lambdas.insert(0, "pool", pool_name)
            lambdas.insert(1, "experts", "+".join(experts))
            lambda_rows.append(lambdas)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(out_root / "stage5i_ab1_expert_contribution_mean_std.csv", index=False)
    if lambda_rows:
        pd.concat(lambda_rows, ignore_index=True).to_csv(
            out_root / "stage5i_ab1_selected_lambdas.csv", index=False
        )

    completeness = []
    for pool_name in pools:
        pool_root = out_root / pool_name / "stage5i"
        complete = all((pool_root / f"seed{seed}" / "target_query_predictions.csv").exists() for seed in seeds)
        completeness.append({"pool": pool_name, "complete": complete, "stage5i_dir": str(pool_root)})
    pd.DataFrame(completeness).to_csv(out_root / "stage5i_ab1_completeness.csv", index=False)


def copy_archive(out_root: Path, archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    for name in [
        "stage5i_ab1_expert_contribution_mean_std.csv",
        "stage5i_ab1_selected_lambdas.csv",
        "stage5i_ab1_completeness.csv",
        "README.md",
    ]:
        src = out_root / name
        if src.exists():
            shutil.copy2(src, archive_root / name)


def stage5i_complete(stage5i_root: Path, seeds: list[int]) -> bool:
    if not (stage5i_root / "stage5i_mean_std.csv").exists():
        return False
    return all(
        (stage5i_root / f"seed{seed}" / "target_query_predictions.csv").exists()
        and (stage5i_root / f"seed{seed}" / "selection_lock.json").exists()
        for seed in seeds
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-stage5e-root", type=Path, default=Path("experiments/new4+harp/stage5e_paper_exact_seed1_5"))
    parser.add_argument("--hmoe-root", type=Path, default=Path("experiments/new4+harp/hmoe_paper_exact_baseline"))
    parser.add_argument("--out-root", type=Path, default=Path("experiments/new4+harp/stage5i_ab1_expert_contribution"))
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--seeds", default="1 2 3 4 5")
    parser.add_argument("--pools", default="all")
    parser.add_argument("--lambda-grid", default="0 0.05 0.1 0.2 0.3")
    parser.add_argument("--min-support-rel-gain", default="0.05")
    parser.add_argument("--one-se-tolerance", default="0.02")
    parser.add_argument("--metric-batch-size", default="64")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    pools = pool_definitions()
    if args.pools != "all":
        wanted = [item.strip() for item in args.pools.replace(",", " ").split() if item.strip()]
        pools = {name: pools[name] for name in wanted}

    args.out_root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    train_stage5e = scripts / "train_stage5e_design_mass_anchorfree.py"
    train_stage5i = scripts / "train_stage5i_hmoe_shrinkage.py"

    for pool_name, experts in pools.items():
        pool_root = args.out_root / pool_name
        stage5e_root = pool_root / "stage5e"
        stage5i_root = pool_root / "stage5i"
        stage5e_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            seed_base = args.base_stage5e_root / f"seed{seed}"
            catalog = read_candidate_catalog(seed_base)
            specs = candidate_specs(catalog, experts)
            stage5e_seed = stage5e_root / f"seed{seed}"
            stage5e_seed.mkdir(parents=True, exist_ok=True)
            stage5e_cfg = stage5e_args_from_lock(seed_base)
            if not (args.skip_existing and (stage5e_seed / "target_query_predictions.csv").exists()):
                cmd = [
                    sys.executable,
                    str(train_stage5e),
                    "--candidates",
                    *specs,
                    "--out-dir",
                    str(stage5e_seed),
                    "--scopes",
                    *stage5e_cfg["scopes"].split(),
                    "--alpha-grid",
                    stage5e_cfg["alpha_grid"],
                    "--cv-seeds",
                    stage5e_cfg["cv_seeds"],
                    "--support-val-per-kernel",
                    stage5e_cfg["support_val_per_kernel"],
                    "--selection-threshold-grid",
                    stage5e_cfg["selection_threshold_grid"],
                    "--weight-floor-grid",
                    stage5e_cfg["weight_floor_grid"],
                    "--metric-batch-size",
                    args.metric_batch_size,
                    "--method-id",
                    "FSS-R4-AB1-E",
                    "--method-name",
                    f"Stage5I-AB1-Stage5E-{pool_name}",
                    "--artifact-prefix",
                    "stage5e",
                    "--prediction-model",
                    f"stage5e_ab1_{pool_name}",
                    "--outer-seed",
                    str(seed),
                ]
                run_command(cmd, args.dry_run)

        if not (args.skip_existing and stage5i_complete(stage5i_root, seeds)):
            cmd = [
                sys.executable,
                str(train_stage5i),
                "--hmoe-root",
                str(args.hmoe_root),
                "--stage5e-root",
                str(stage5e_root),
                "--out-root",
                str(stage5i_root),
                "--seeds",
                " ".join(str(seed) for seed in seeds),
                "--lambda-grid",
                args.lambda_grid,
                "--min-support-rel-gain",
                args.min_support_rel_gain,
                "--one-se-tolerance",
                args.one_se_tolerance,
                "--metric-batch-size",
                args.metric_batch_size,
            ]
            run_command(cmd, args.dry_run)

    if not args.dry_run:
        collect_summary(args.out_root, pools, seeds)
        readme = f"""# Stage5I-AB1 Expert Contribution Ablation

Purpose: test whether each new GNN expert contributes after anchoring the
prediction on complete HMoE.

Protocol: `paired_shared_hmoe_kmeans_support`, seeds `{seeds}`.

For each expert pool, this launcher reruns Stage5E on support only and then
reruns Stage5I shrinkage:

```text
y = y_hmoe + lambda_target * (y_stage5e_pool - y_hmoe)
```

The lambda grid and support-only selection rules match the main Stage5I run.
Leave-one-out pools are refit from scratch; no query labels are used for
selecting weights, masks, lambdas, or pools.

Pools:

{chr(10).join(f'- `{name}`: {", ".join(experts)}' for name, experts in pools.items())}
"""
        (args.out_root / "README.md").write_text(readme, encoding="utf-8")
        if args.archive_root is not None:
            copy_archive(args.out_root, args.archive_root)

    print(f"Wrote Stage5I-AB1 outputs to: {args.out_root}")


if __name__ == "__main__":
    main()
