from pathlib import Path
import argparse
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRAIN = HERE / "train_sgformer_rgcn_hmoe_protocol.py"


def run(command, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--source-epochs", type=int, default=5)
    parser.add_argument("--target-epochs", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=ROOT / "hares_release/artifacts/results/sgformer_rgcn_qor/batchsafe_pilot")
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()

    cache_root = args.cache_root
    if cache_root is None:
        cache_candidates = list((ROOT / "HierarchicalMoE-master/save/programl").glob("v21_MLP-True*"))
        if len(cache_candidates) != 1:
            raise ValueError(f"Expected one HMoE cache root, found: {cache_candidates}")
        cache_root = cache_candidates[0]

    output_root = args.output_root.resolve()
    common = [
        "--cache-root", str(cache_root),
        "--k", "50",
        "--batch_size", "64",
        "--hidden_dim", "64",
        "--rgcn_layers", "2",
        "--sgformer_layers", "1",
        "--num_heads", "1",
        "--num_relations", "3",
        "--global_dim", "21",
        "--predictor-mode", "shared",
        "--loss_name", "hmoe_sum_mse",
        "--progress", "off",
    ]
    for seed in args.seeds:
        manifest = ROOT / f"hares_release/artifacts/protocol/manifest_kmeans_hmoe_source_exact_seed{seed}.csv"
        seed_root = output_root / f"seed{seed}"
        source_dir = seed_root / "source"
        target_dir = seed_root / "support_ft"
        source_ckpt = source_dir / "model.pt"
        if not source_ckpt.exists():
            command = [
                sys.executable, str(TRAIN), "--phase", "source", "--manifest", str(manifest),
                "--output-dir", str(source_dir), "--seed", str(seed),
                "--epochs", str(args.source_epochs), "--patience", "3", "--batch_size", "64",
                *common,
            ]
            print(f"seed{seed}: source training", flush=True)
            run(command, seed_root / "source.log", ROOT)
        if not (target_dir / "target_query_predictions.csv").exists():
            command = [
                sys.executable, str(TRAIN), "--phase", "support_finetune", "--manifest", str(manifest),
                "--output-dir", str(target_dir), "--source-checkpoint", str(source_ckpt),
                "--seed", str(seed), "--epochs", str(args.target_epochs), "--patience", "5",
                "--eval-every", "10", *common,
            ]
            print(f"seed{seed}: target fine-tuning", flush=True)
            run(command, seed_root / "support_ft.log", ROOT)
        print(f"seed{seed}: complete", flush=True)

    command = [
        sys.executable, str(HERE / "summarize_protocol_results.py"),
        "--run-root", str(output_root), "--output-dir", str(output_root / "summary"),
        "--seeds", *(str(seed) for seed in args.seeds),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
