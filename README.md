# HARES reproducibility release

This repository contains the prediction-level HARES implementation used for
the final HMoE-anchored residual experiment. It intentionally does not include
the experiment reports, logs, generated summaries, HMoE source checkout, raw
HLS cache, or model checkpoints.

## Reproduction scope

The release reproduces the HARES QoR fusion from exported prediction artifacts.
It does not retrain the upstream HMoE or the four heterogeneous GNN experts.
Those components must be obtained from the official HMoE release or supplied as
prediction artifacts under `artifacts/`.

The fixed protocol is recorded in
`configs/final_protocol.json`: six target kernels, 50 support designs per
kernel, 837 query designs, and QoR seeds 1--5. The shrinkage coefficient is
selected from the support labels only.

## Mapping to the attached paper

The paper names the local expert adapters GPS-Lite, Attn-VN, DepthMix, and
GatedGCN. The corresponding frozen artifact names here are `graphgps`,
`exphormer`, `polynormer`, and `gatedgcn`. These are protocol-aligned local
adapters, not claims of complete reproductions of the original systems.

The internal labels `Stage5E-DMW-AFRC`/`FSS-R2-M1` refer to the paper's
coverage-weighted convex correction, and `Stage5I`/`FSS-R4-M1` refers to the
metric-wise residual shrinkage stage. They are provenance labels, not extra
methods reported in the paper.

## External SOTA comparison scope

The external comparison table places HMoE, HARES, MPM, HGBO-HGP, GNNDSE-QoR,
IronManPro-GPP, and SGFormer in the same SOTA comparison scope. HARES uses the
fixed five-expert correction pool `HARP + GraphGPS + Exphormer + Polynormer +
GatedGCN`; SGFormer is an external SOTA baseline and is deliberately excluded
from that pool. The included SGFormer-RGCN files are a protocol-aligned adapter
under the HMoE manifest/cache split, not a claim of a complete reproduction of
the original SGFormer dataset or feature pipeline.

The attached PDF reports sample-weighted Total MSE: HMoE `0.14111 ± 0.01434`
and HARES `0.12176 ± 0.01518`. The same frozen predictions also yield the
HMoE-compatible batch-mean loss `0.143466 ± 0.014127` and `0.120705 ±
0.013139`, respectively. These are two different aggregations of the same
predictions; the paper's Table I uses the sample-weighted Total MSE.

The paper's headline Total MSE is the sample-weighted sum of the five target
MSE values across all 837 query designs. The included Stage5I script also
reports the HMoE-compatible `hmoe_test_loss_batch_mean`: for every batch, the
five target MSE values are summed and the batch losses are averaged with equal
batch weight (`metric_batch_size=64`). Do not substitute one for the other.

## Install

```bash
python -m venv .venv
# Activate the environment using the shell appropriate for your system.
python -m pip install -r requirements.txt
```

Run commands from the repository root so that the relative artifact paths are
resolved consistently.

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
├─ sota/sgformer/seed{1..5}/
│  ├─ target_support_predictions.csv
│  └─ target_query_predictions.csv
└─ candidates/seed{1..5}/<model>/
   ├─ target_support_predictions.csv
   └─ target_query_predictions.csv
```

Protocol provenance is also included under `artifacts/protocol/`: the five
HMoE KMeans manifests and path-sanitized per-seed audit JSON files. The compact
`artifacts/manifests/seed*_ids.csv` files provide a quick support/query ID
check. `artifacts/input_checksums.csv` records the SHA-256 of every frozen
input artifact.

The four heterogeneous candidate names are `graphgps`, `exphormer`,
`polynormer`, and `gatedgcn`; the anchor candidate is
`harp_hmoe_support_paired`. The external SOTA input is `sota/sgformer` and is
validated against the same per-seed support/query IDs. Column requirements are
in `schemas/artifact_contract.json`.

Prediction CSVs and HMoE data may have separate redistribution terms. If they
cannot be placed in this Git repository, publish them as a versioned external
artifact and keep the same relative layout and SHA-256 index.

The full protocol manifests under `artifacts/protocol/` contain source and
target IDs, pragma keys, split assignments, and ground-truth QoR columns. They
are included for pairing and audit, but their redistribution still depends on
the upstream HMoE/HLSyn data terms.

## Run HARES from locked Stage5E artifacts

```bash
python scripts/train_stage5i_hmoe_shrinkage.py \
  --hmoe-root artifacts/inputs/hmoe \
  --stage5e-root artifacts/inputs/stage5e \
  --out-root outputs/stage5i \
  --seeds "1 2 3 4 5"
```

The script writes regenerated predictions, support-only lambda searches,
selection locks, and per-seed metrics under `outputs/`. These generated files
are ignored by Git.

To reproduce the sample-weighted Total MSE used in the paper's Table I:

```bash
python scripts/reproduce_paper_metrics.py --hares-root outputs/stage5i
```

This prints the HARP, HMoE, unshrunk pool, and HARES rows from the frozen
prediction artifacts. The SGFormer external baseline inputs are included under
`artifacts/inputs/sota/sgformer`; other external baseline rows require their
own prediction artifacts.

## Regenerate Stage5E before HARES

Each candidate directory can be passed to the Stage5E script. For one seed:

```bash
python scripts/train_stage5e_design_mass_anchorfree.py \
  --candidates \
    harp_hmoe_support_paired=artifacts/inputs/candidates/seed1/harp_hmoe_support_paired \
    graphgps=artifacts/inputs/candidates/seed1/graphgps \
    exphormer=artifacts/inputs/candidates/seed1/exphormer \
    polynormer=artifacts/inputs/candidates/seed1/polynormer \
    gatedgcn=artifacts/inputs/candidates/seed1/gatedgcn \
  --out-dir outputs/stage5e/seed1 \
  --outer-seed 1 \
  --method-id FSS-R2-M1 \
  --method-name Stage5E-DMW-AFRC
```

After all five Stage5E seeds are generated, pass
`--stage5e-root outputs/stage5e` to the HARES command. The ablation launcher
is `scripts/run_stage5i_expert_contribution_ablation.py`.

The Stage5E script expects its output directory to exist. Create it before the
command above, for example:

```bash
mkdir -p outputs/stage5e/seed1
```

The paper release covers QoR prediction and the reported expert-pool ablation.
Other internal exploration branches are intentionally excluded, including
dataset-contained DSE-proxy ranking, OOF/cross-fit branches, and generic
oracle/router diagnostics. The attached PDF contains DSE as background, but it
does not report a dataset-contained DSE proxy; newer manuscript variants should
publish that analysis as a separate artifact if it is retained.

`release_core.py` and `release_metrics.py` are implementation helpers for
loading candidate predictions, fitting the coverage-weighted correction,
computing the paper metrics, and writing locks. They are not extra methods
reported in the paper.

## Provenance and licensing

The HMoE implementation, HLSyn-derived data, checkpoints, and any external
expert implementations remain upstream dependencies. Check their licenses
before redistributing them. `CITATION.cff` is included; choose and add a
license for the HARES-authored code before creating the public GitHub release.
