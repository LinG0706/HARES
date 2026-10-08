# SGFormer-RGCN QoR baseline under the HMoE protocol

This adapter ports the SGFormer global-attention and RGCN local-relation
backbone from `E:\idea\tang` to the HMoE manifest/cache protocol. It uses
the HMoE source split, 50 K-means-selected fine-tuning designs per target
kernel, the same left-out designs, and the five HMoE QoR targets.

The current table result uses `--predictor-mode shared`, where one
SGFormer-RGCN graph backbone feeds five target heads. The strict
`--predictor-mode independent` option is also available but is much more
expensive on CPU. The original Tang implementation expects a different
dataset and a six-target HLS-report feature vector. Those inputs are not
available in the HMoE cache, so this result is a protocol-aligned
SGFormer-RGCN adapter rather than a direct reproduction of the original
Tang experiment.

The frozen five-seed external-baseline inputs are published at
`artifacts/inputs/sota/sgformer/seed{1..5}/`. Each seed contains
`target_support_predictions.csv` (300 rows) and `target_query_predictions.csv`
(837 rows). `scripts/validate_release_inputs.py` checks their file presence,
row counts, protocol IDs, and SHA-256 entries. These files are comparison
artifacts only; SGFormer is not part of the HARES correction pool.
