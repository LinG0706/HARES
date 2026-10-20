# Expert fusion and HMoE-anchored shrinkage

This snapshot reproduces the final paper's ablation table from the bundled
five-seed prediction artifacts. Run from the release root:

```bash
python scripts/run_expert_fusion_ablation.py --output-dir outputs/ablation
```

| Expert fusion | Shrinkage | Total MSE (mean +/- sample SD) |
| --- | --- | --- |
| HMoE only | -- | 0.1411 +/- 0.0143 |
| Uniform pool | No | 0.1519 +/- 0.0382 |
| Learned pool | No | 0.1459 +/- 0.0328 |
| Uniform pool | Yes | 0.1288 +/- 0.0200 |
| Learned pool (HARES) | Yes | 0.1218 +/- 0.0152 |

The uniform pool assigns weight 1/5 to each of HARP, GraphGPS, Exphormer,
Polynormer, and GatedGCN, without fitting or expert selection. The learned
pool replays the main experiment's support-fitted weights and expert masks.
The direct rows use their pool predictions without an HMoE anchor. The
anchored rows use `anchor + lambda_target * (pool - anchor)`. Both select
coefficients from `[0, 0.05, 0.10, 0.20, 0.30]` using the same support-only
5% gain threshold and 2% tolerance rule. Every requested seed's coefficients
are locked before query evaluation.

For each seed, all methods use the same 300 support designs (50 per target
kernel) and 837 query designs. Per-target MSE is the mean squared error over
all query designs; Total MSE is the sum over the five targets. Mean and
sample standard deviation (`ddof=1`) are calculated over the five seed-level
metrics, including the Total MSE standard deviation.

Files:

- `per_seed.csv` and `mean_std.csv`: unrounded results for all five rows.
- `ablation_table.tex`: the final paper's five-row layout and model names.
- `seed*/selection_lock.json`: support-only coefficient selections and hashes.
- `seed*/uniform_support_predictions.csv` and `learned_support_predictions.csv`:
  the support predictions used by the shrinkage selectors.
- `seed*/*_lambda_search.csv`: complete support shrinkage searches.
- `seed*/<variant>/target_query_predictions.csv`: query exports for all five rows.
- `all_support_selections_locked.json`, `definition.json`, `input_checksums.csv`,
  and `completion.json`: analysis definition, selection locks, and provenance.

The variants are `anchor_only`, `uniform_direct`, `learned_direct`,
`uniform_anchored`, and `learned_anchored`. The last is the same HARES
prediction reported in the main table.

The design was developed after the main experiments; no query labels are used
to choose its fitted weights, masks, or shrinkage coefficients. Anchoring
plus shrinkage is tested jointly, and learned weighting plus expert selection
is tested jointly. This table does not independently establish an accuracy
benefit for coverage weighting, convexity constraints, or metric-wise versus
shared shrinkage. No neural predictor is retrained or selected using query MSE.
