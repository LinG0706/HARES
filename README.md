# HARES reproducibility release

This repository contains the prediction-level HARES implementation used for
the HARES QoR experiments. It reproduces the HMoE-anchored residual ensemble
from frozen prediction artifacts and does not retrain the upstream HMoE model
or the five heterogeneous GNN predictors.

## Reproduction scope

The release reproduces the HARES QoR fusion from exported prediction artifacts.
The correction pool contains five independently trained predictors named in the
paper: `HARP`, `GraphGPS`, `Exphormer`, `Polynormer`, and `GatedGCN`. HMoE
predictions provide the anchor, while the five predictor outputs are combined
through support-only convex fitting and metric-wise residual shrinkage.

The fixed protocol is recorded in `configs/final_protocol.json`: six target
kernels, 50 support designs per kernel, 837 query designs, and QoR seeds 1--5.
The shrinkage coefficient is selected from support labels only.

## Mapping to the paper

The portable candidate directories use the paper's model names in lowercase:

```text
harp/
graphgps/
exphormer/
polynormer/
gatedgcn/
```

These five predictors form the HARES correction pool. `HMoE` is supplied as the
anchor prediction under `artifacts/inputs/hmoe/`. The five-seed query exports
for `GNN-DSE`, `IronMan-Pro`, `SGFormer`, and `MPM` are included under
`artifacts/inputs/baselines/gnndse/`, `artifacts/inputs/baselines/ironman/`,
`artifacts/inputs/baselines/sgformer/`, and `artifacts/inputs/baselines/mpm/`.
They are comparison methods, excluded from the HARES correction pool.
Model names identify the methods compared under this protocol; exporting
predictions does not reproduce the original papers' training experiments.

All four comparison methods use the same 50 support designs per target kernel
and the same 837 query designs for each of the five paired seeds. Like GNN-DSE
and IronMan-Pro, MPM and SGFormer are supplied as frozen query CSVs and export
metadata, not as neural-network training scripts or checkpoints.
MPM uses graph/text fusion with deterministic soft graph gates (temperature
0.5) and pragma-prefixed CodeBERT features. SGFormer uses protocol-aligned graph
inputs and global L2 gradient clipping at 1.0 during source and support training.
These implementation settings are recorded in each `prediction_export.json`;
the exports are protocol-aligned comparisons, not unchanged reproductions of
the external systems.

The internal labels `Stage5E-DMW-AFRC`/`FSS-R2-M1` refer to the
coverage-weighted convex correction, and `Stage5I`/`FSS-R4-M1` refers to the
metric-wise residual shrinkage stage. They are provenance labels, not extra
models reported in the paper.

The paper reports sample-weighted Total MSE: the sum of the five target MSEs
computed over all 837 query designs. The Stage5I script also reports the
HMoE-compatible batch-mean diagnostic, in which the five target MSEs are summed
within each batch and the batch losses are averaged with equal batch weight.
These are different aggregations of the same predictions; the paper table uses
the sample-weighted Total MSE.

## Install

```bash
python -m venv .venv
# Activate the environment using the shell appropriate for your system.
python -m pip install -r requirements.txt
```

Run commands from the repository root so that relative artifact paths resolve
consistently.

Validate the portable input bundle before running the experiment:

```bash
python scripts/validate_release_inputs.py
```

## Expected artifacts

The input layout is:

```text
artifacts/inputs/
├─ hmoe/seed{1..5}/target_finetune/
│  ├─ target_support_predictions.csv
│  └─ target_query_predictions.csv
├─ stage5e/seed{1..5}/
│  ├─ selection_lock.json
│  ├─ stage5e_convex_weights.csv
│  └─ target_query_predictions.csv
├─ baselines/<model>/seed{1..5}/
│  ├─ target_query_predictions.csv
│  └─ prediction_export.json
└─ candidates/seed{1..5}/<model>/
   ├─ target_support_predictions.csv
   └─ target_query_predictions.csv
```

Protocol provenance is included under `artifacts/protocol/`. The compact
`artifacts/manifests/seed*_ids.csv` files provide a quick support/query ID
check. `artifacts/input_checksums.csv` records the SHA-256 of every frozen
input artifact.

The candidate names are `harp`, `graphgps`, `exphormer`, `polynormer`, and
`gatedgcn`. Column requirements are defined in
`schemas/artifact_contract.json`.

Prediction CSVs and HMoE data may have separate redistribution terms. If they
cannot be placed in this Git repository, publish them as a versioned external
artifact and keep the same relative layout and SHA-256 index.

## Run HARES from locked Stage5E artifacts

```bash
python scripts/train_stage5i_hmoe_shrinkage.py \
  --hmoe-root artifacts/inputs/hmoe \
  --stage5e-root artifacts/inputs/stage5e \
  --out-root outputs/stage5i \
  --seeds "1 2 3 4 5"
```

To reproduce the sample-weighted Total MSE used in the paper:

```bash
python scripts/reproduce_paper_metrics.py \
  --hares-root outputs/stage5i \
  --output-dir outputs/paper
```

This recomputes all eleven table rows, checks seed-specific query IDs and labels,
and writes per-seed metrics, five-seed means/sample standard deviations, and
`main_table.tex`. The upper section follows the final paper order: GatedGCN,
HARP, GraphGPS, Polynormer, Exphormer, HMoE, and HARES. The lower section contains
IronMan-Pro, GNN-DSE, SGFormer, and MPM in descending mean Total MSE order.
A verified snapshot is included in `artifacts/paper/`. The learned pool without
shrinkage is reported in the ablation table only. No baseline is refitted by
this command. All evaluation labels and predictions retain their original values.

## Regenerate the convex correction before HARES

For one seed:

```bash
python scripts/train_stage5e_design_mass_anchorfree.py \
  --candidates \
    harp=artifacts/inputs/candidates/seed1/harp \
    graphgps=artifacts/inputs/candidates/seed1/graphgps \
    exphormer=artifacts/inputs/candidates/seed1/exphormer \
    polynormer=artifacts/inputs/candidates/seed1/polynormer \
    gatedgcn=artifacts/inputs/candidates/seed1/gatedgcn \
  --out-dir outputs/stage5e/seed1 \
  --outer-seed 1 \
  --method-id FSS-R2-M1 \
  --method-name Stage5E-DMW-AFRC
```

Create the output directory before running the command:

```bash
mkdir -p outputs/stage5e/seed1
```

After all five Stage5E seeds are generated, pass
`--stage5e-root outputs/stage5e` to the HARES command.

## Reproduce the paper ablation

Run the full expert-fusion/shrinkage comparison from the bundled inputs:

```bash
python scripts/run_expert_fusion_ablation.py --output-dir outputs/ablation
```

The five rows match the final manuscript:

| Expert fusion | Shrinkage | Total MSE (mean +/- sample SD) |
| --- | --- | --- |
| HMoE only | -- | 0.1411 +/- 0.0143 |
| Uniform pool | No | 0.1519 +/- 0.0382 |
| Learned pool | No | 0.1459 +/- 0.0328 |
| Uniform pool | Yes | 0.1288 +/- 0.0200 |
| Learned pool (HARES) | Yes | 0.1218 +/- 0.0152 |

The uniform pool assigns weight 1/5 to every expert. The learned pool replays
the same support-fitted weights and expert masks as the main HARES experiment.
Without shrinkage, each row directly uses its pool prediction. With shrinkage,
the prediction is `anchor + lambda_target * (pool - anchor)` using the same
support-only selection rule for both pools. All five seeds' coefficients are
locked before query evaluation. No upstream predictor is retrained.

Outputs include per-seed predictions for all five rows, per-seed and aggregate
MSE CSVs, support-selection locks, input hashes, and `ablation_table.tex`.
The verified snapshot is in `artifacts/ablation/`. This is a matched two-factor
diagnostic designed after the main experiments. It tests anchoring plus
shrinkage jointly and learned fitting plus expert selection jointly; it does
not establish independent accuracy gains for every internal component.

The paper release covers QoR prediction and the expert-fusion/shrinkage ablation.
Internal exploration branches, generated logs, model checkpoints, and generic
oracle/router diagnostics are intentionally excluded.

`release_core.py` and `release_metrics.py` are implementation helpers for
loading candidate predictions, fitting the coverage-weighted correction,
computing the paper metrics, and writing locks. They are not extra models
reported in the paper.

## Provenance and licensing

The HMoE implementation, HLSyn-derived data, checkpoints, and expert
implementations remain upstream dependencies. Check their licenses before
redistributing them. `CITATION.cff` is included; choose and add a license for
the HARES-authored code before creating the public GitHub release.
