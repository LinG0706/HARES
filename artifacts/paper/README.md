# Paper metrics

These files were recomputed from paired query prediction CSVs for seeds 1--5.
For each seed and target, the MSE is the mean squared error over all 837 query
designs. Total MSE is the sum of the five target MSEs. Standard deviations are
sample standard deviations across seed-level metrics (`ddof=1`). The Total MSE
standard deviation is computed from seed-level totals, never by adding the
five target standard deviations.

`per_seed.csv` preserves the seed-level results; `mean_std.csv` preserves
unrounded summary values; `main_table.tex` is generated from these values.
The row order follows the final manuscript. The lower comparison section is
IronMan-Pro, GNN-DSE, SGFormer, and MPM, sorted by descending mean Total MSE.

Regenerate after replaying HARES:

```bash
python scripts/reproduce_paper_metrics.py --output-dir artifacts/paper
```

The eleven-method snapshot includes the final five-seed MPM and SGFormer
query exports. MPM reports Total MSE `0.1552 +/- 0.0506`; SGFormer reports
`0.1919 +/- 0.0523`, using the same sample-weighted evaluation. Their frozen
implementation settings are recorded in each `prediction_export.json`.
The learned pool without shrinkage is included only in the ablation snapshot
under `artifacts/ablation/`. It is no longer a row of this main table.
Existing methods retain their unrounded metrics. Three per-target entries in the
previously supplied manuscript differ at the final displayed digit from
direct rounding of the released predictions:

| Entry | Manuscript | Direct rounding of exports |
| --- | --- | --- |
| HMoE BRAM standard deviation | 0.0007 | 0.0006 |
| HARES DSP mean | 0.0345 | 0.0344 |
| HARES BRAM standard deviation | 0.0007 | 0.0006 |

The generated table uses direct rounding of the exports; it does not alter
predictions to match manually entered cells.
