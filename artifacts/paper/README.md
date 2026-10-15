# Paper metrics

These files were recomputed from paired query prediction CSVs for seeds 1--5.
For each seed and target, the MSE is the mean squared error over all 837 query
designs. Total MSE is the sum of the five target MSEs. Standard deviations are
sample standard deviations across seed-level metrics (`ddof=1`). The Total MSE
standard deviation is computed from seed-level totals, never by adding the
five target standard deviations.

`per_seed.csv` preserves the seed-level results; `mean_std.csv` preserves
unrounded summary values; `main_table.tex` is generated from these values.
The comparison rows are sorted by descending mean Total MSE.

Regenerate after replaying HARES:

```bash
python scripts/reproduce_paper_metrics.py --output-dir artifacts/paper
```

The supplied manuscript and the recomputed exports agree on all ten displayed
Total MSE means and standard deviations. Three per-target entries in the
supplied manuscript differ at the final displayed digit from direct rounding
of the released predictions:

| Entry | Manuscript | Direct rounding of exports |
| --- | --- | --- |
| HMoE BRAM standard deviation | 0.0007 | 0.0006 |
| HARES DSP mean | 0.0345 | 0.0344 |
| HARES BRAM standard deviation | 0.0007 | 0.0006 |

The generated table uses direct rounding of the exports; it does not alter
predictions to match manually entered cells.
