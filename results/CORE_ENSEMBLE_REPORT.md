# Full-length core ensemble results

## Scope and run settings

The core comparison includes MM32 and BB170, each with a no-polymer control and P1, P2, or P3 at 7,000 a.u. Each strain-condition pair has 10 independent seeds: 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, and 2021. All runs reached 30,000 a.u.; outputs were sampled every 1,000 a.u. This produced 80 replicates, 2,480 replicate-time rows, 80 checkpoints, and 24 plots.

## Solver and interpretation limits

To make this full-length ensemble tractable, `run_long_core.py` uses a 100-a.u. operator-split tau-leap. Signal and polymer movement and binding updates are batched, and each bacterial cluster can move at most once per leap. This is an approximate stochastic model, not the paper's exact direct-Gillespie algorithm. Polymer concentration is represented as 100 a.u. per point-agent, which remains an unverified mapping assumption. The run covers the core P1/P2/P3 comparisons at 7,000 a.u.; it does not cover every polymer concentration or the 25-pair affinity grid.

Treat the plotted trends as exploratory. The leap size may change binding, transport, and cluster dynamics; endpoint similarity to the no-polymer condition cannot be interpreted as a definitive failure to reproduce the paper until the implementation choices are validated against exact short runs and the published parameter mapping is resolved.

## Endpoint at 30,000 a.u.

Values below are mean ± sample SD across the 10 seeds. Fold change is normalized to the paired same-strain, same-seed no-polymer control at the same time.

| Strain | Condition | Mean luminescence ± SD | Fold change ± SD |
|---|---|---:|---:|
| MM32 | no polymer | 156,243 ± 9,218 | 1.000 ± 0.000 |
| MM32 | P1 | 154,231 ± 12,992 | 0.987 ± 0.047 |
| MM32 | P2 | 154,878 ± 11,059 | 0.991 ± 0.040 |
| MM32 | P3 | 154,835 ± 11,263 | 0.991 ± 0.038 |
| BB170 | no polymer | 170,620 ± 6,389 | 1.000 ± 0.000 |
| BB170 | P1 | 169,348 ± 11,347 | 0.993 ± 0.062 |
| BB170 | P2 | 173,389 ± 12,073 | 1.016 ± 0.057 |
| BB170 | P3 | 168,070 ± 10,359 | 0.985 ± 0.048 |

These results show only small differences from controls relative to between-seed variability under this coarse approximation. They are not directly comparable to published values without resolving the caveats above.

## Outputs

- `configuration.json`: explicit seeds, horizon, step size, and scope.
- `replicate_results.csv`: raw time-series data for all independent runs.
- `ensemble_summary.csv`: timewise means and standard deviations.
- `checkpoints/`: one CSV per completed replicate.
- `report_figures/`: curated 3-figure set for a concise report: one two-panel time-course comparison per strain, plus a pooled endpoint fold-change comparison.
- `report_figures/figure_legends.md`: captions and interpretation note for the three curated figures.
- `plots/`: all 24 individual per-condition plots, retained for detailed inspection.

Regenerate the curated figures with `python outputs/plot_report_figures.py`. The run can be repeated or resumed with `python outputs/run_long_core.py`; existing completed checkpoints are reused.
