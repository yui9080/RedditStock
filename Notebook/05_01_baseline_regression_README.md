# Baseline regressions

Both baseline period samples use one entry point:

```powershell
python Notebook/05_01_baseline_regression.py
python Notebook/05_01_baseline_regression.py --sample baseline
python Notebook/05_01_baseline_regression.py --sample 2019_2020
```

The default runs both samples. `baseline` uses 2014-06-01 through 2020-06-30;
`2019_2020` uses available 2019/2020 network months within 2019-01-01 through
2020-12-31 (currently through 2020-09). Both use the same regression code.
The old 05_02 baseline wrapper has been removed. All baseline outputs now use
the 05_01 prefix and share Data/process/05_01_baseline_regression/<UTC time>/,
with baseline/ and 2019_2020/ subdirectories. Existing historical outputs remain.
05_02_degree_group_regression.py is now the degree-group regression entry point
and saves to Data/process/05_02_degree_group_regression/<UTC time>/.

The newest completed 05 panel is used. Optional `--panel`, `--start`, `--end`,
`--output-dir` overrides exist. Date overrides apply to every selected sample;
the 2019/2020 network-year restriction still applies. The selected sample subdirectories are always created inside the run directory,
including when an explicit --output-dir is supplied.
The next trading return may be after the predictor-date period end, as already
defined in the panel. Network filter exclusions are separate from date
restrictions and complete-case exclusions. 05_02 continues to reuse the same
estimation functions, without running either sample on import.

Dependencies: pandas, numpy, scipy, pyarrow, linearmodels. Project-local copies
in `.tmp_regression_deps` are automatically used when present. They can be
installed with `python -m pip install --target .tmp_regression_deps linearmodels`.

Each model uses its own complete cases. There is no additional ever-connected
or full-period-survival stock filter: valid reddit_ret already requires usable
network peers on that date. BM, industry return and other unrequested controls
are not included. Model 3 is Model 2 + illiquidity only. idio_vol and log_me are excluded,
so their missingness does not restrict any baseline sample. The degree-group
05_02 script now estimates all three specifications. If a requested model has
no complete cases it is reported unavailable without inventing another model.

Day FE are absorbed by linearmodels AbsorbingLS/HDFE; no manual demeaning, firm
FE or weighting is performed. An intercept is included. One-way clustered
covariance uses PERMNO with CR1 finite-sample correction including the absorbed
day FE degrees of freedom (G/(G-1) × (N-1)/(N-K_total));
reported p-values use Student t with G-1 degrees of freedom. Overall R2 and
adjusted R2 include absorbed date effects, not just the within-date regression.

All five return variables, including the dependent return, are multiplied by
100. Illiquidity retains its original panel scale (raw Amihud measure times 1e6). CSV
`coefficient` is the **actual fitted slope**. `coefficient_times_100` is an
additional reporting scale, not a transformed regressor or a different model.
With both returns in percentage points, slope 0.04 means a 1 pp peer move is
associated with a 0.04 pp focal move, i.e. 4% of that movement. If the reported
coefficient is 4 (=100×0.04), dividing that report value by 100 yields the slope.
The supplied example claiming actual percent/percent slope 4 means 0.04 pp is
inconsistent: the actual slope must not be divided by 100 a second time.

The table reports actual slopes, with t-statistics below each coefficient and
stars based on p<.01/.05/.10. Its Model 3 column is NA when unestimated.
Model 3 diagnostics/correlations are likewise explicitly unavailable rather
than being substituted with Model 2's sample. Descriptive paper comparison
uses the user's positive-coefficient reference, not an independent replication
of the original paper. Coefficients are associations, not causal carryover.

Duplicate PERMNO-date raises an error. Missing/nonfinite values generate
warnings and model-specific exclusion counts; there is no zero filling,
interpolation, winsorization, outlier trimming or standardization. Zero variance
warns and stops estimation rather than silently dropping the control. High
pairwise correlations (>0.9 absolute) are warned in the Model 3 diagnostic sample.
