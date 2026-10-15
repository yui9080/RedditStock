# 05 daily regression panel

## Unified sample dates

`regression_samples.py` defines full_sample = 2012-11-01 to 2020-08-31 and
late_sample = 2019-01-01 to 2020-08-31, inclusive, for 05 through 05_03.
05 writes the full-sample date range by default; downstream scripts select the
late subset. Optional --start/--end are explicit overrides recorded in metadata.
The complete raw CRSP history remains available while calculating lagged
returns, prior-month ME and industry returns; only output dates are restricted.
Rows excluded by this date restriction are counted separately. ret_t1 on the
last predictor date can refer to the next trading observation after 2020-08-31.

Run `python Notebook/05_build_regression_panel.py`. The program uses the newest
completed **native PERMNO** network under `04_monthly_ticker_network`, full
`Data/raw/stock/RAW_crsp.csv`, and the effective SIC history in `RAW_name.csv`.
The request calls the network stage 03; in this workspace that stage is 04.
The full CRSP universe is needed for industry peer returns; ticker-based CRSP
exclusions are not used as a substitute for the industry universe.

Each run creates a new `Data/process/05_regression_panel/<UTC time>/` containing:

- `04_regression_panel.parquet`: one row per PERMNO and daily CRSP observation.
- `04_variable_summary.csv`: exact distribution statistics, using linear quantiles.
- `04_yearly_missing_rates.csv`: yearly counts and missing percentages.
- `04_merge_diagnostics.csv`: source, availability and complete-baseline counts.
- `run_summary.json`: inputs, warnings, units, choices, timing and runtime.

There is no outcome regression, winsorization, standardization or missing-return
imputation. All valid CRSP rows inside the requested output period remain,
including stocks outside the Reddit network and rows with missing controls.
Warm-up rows contribute to controls but are not written outside that period.
Date exclusions and invalid stock/date keys are counted separately. Exact duplicates are removed only after every
original CRSP column is reread and verified identical; a warning and dropped-row
count are recorded. Conflicting duplicate keys raise an error.

## Timing and variable definitions

- `network_month` is the observation month. Its graph uses exactly M-6 through
  M-1. Missing graph months are retained with missing Reddit variables; a graph
  is never carried into later months or reconstructed from ticker guesses.
- `reddit_ret` uses the fixed monthly `log1p(n_common_threads)` adjacency and
  the peers' same-day returns, renormalizing over valid peer returns. No focal
  return enters the sum. Zero-valued temporary array masks implement sparse
  algebra, and do not replace missing returns in the panel.
- `n_reddit_peers` is monthly degree; `n_valid_peers_d` counts usable peers that
  day. Isolates have zero peers and missing Reddit return. Stocks outside the
  graph have missing peer counts and `has_network_node=False`. The merge
  diagnostic for a valid network counts rows whose stock is a graph node.
- `ret_t1` is the next row's return in the stock's sorted CRSP trading sequence,
  including Friday-to-Monday transitions. No calendar-day offset is used.
- `ret_20_2` compounds 19 returns, shift 2; `ret_120_21` compounds 100 returns,
  shift 21. Every return in each full window must be valid. A -100% return makes
  the product zero. Numeric CRSP missing codes below -1, alphabetic codes and
  nonfinite returns are missing, not zero. `dlret` is kept separately, never
  combined automatically.
- `me_daily` retains contemporaneous capitalization. `me` and
  `me_lag_monthend` are the last positive, finite daily ME from the **exact
  preceding calendar month**, fixed within the month. `log_me` uses this value.
  PRC×SHROUT is used when available (SHROUT assumed thousand shares); otherwise
  supplied DlyCap is used. Default ME unit is **thousand USD**. `--me-unit`
  declares the supplied DlyCap unit; it does not rescale that source column.
  This declared unit must be checked against the user's CRSP export.
- Effective SIC is assigned by `start <= date <= end`, independently of ticker.
  Conflicting codes are missing with a warning. Industry return uses all other
  same-two-digit-SIC stocks with valid same-day returns and positive ME from
  their own preceding trading observation; the focal stock is excluded.
- `illiquidity_raw` averages abs(RET)/(abs(PRC)×VOL) over prior observations
  d-20 to d-1, minimum 10 valid days. `illiquidity_scaled` is raw×1e6 and
  `illiquidity` is this scaled version. Prices are USD and volume is shares;
  zero/negative dollar volume contributes no valid observation.

**Amihud illiquidity is an implementation choice because Wang and Zhang (2024)
do not provide an explicit formula in the available paper text.**

## Optional fundamentals

Use `--fundamentals PATH` (CSV or Parquet). Required: mapped `permno`, actual
`available_date`, and `measurement_date`, `datadate` or `date`. Fiscal date alone
is not treated as public availability; no filing-date lag is invented and no
GVKEY-to-PERMNO link is guessed. Availability preceding measurement is an error.
Duplicate PERMNO/availability-date versions must be resolved upstream.

If the input contains `bm`, it is used directly only when its measurement is in
the preceding month and it was available by that month end. Older quarterly
ratios are not relabeled as last-month-end BM. A precomputed feed must already
implement its own correct market-equity timing. If the input instead has book
equity ingredients, also supply `--book-equity-unit usd|thousand_usd|million_usd`.
The latest publicly available BE at the previous month end is divided by that
month's ME. Shareholders' equity priority is seqq, ceqq+pstkq, then atq-ltq;
pstkq is subtracted. In the ceqq branch preferred stock algebraically cancels.
Optional missing txditcq contributes zero, as the explicit optional-tax-term
convention. Missing required preferred stock in the other branches prevents BE
construction. BM may be negative; log_bm is missing whenever BM <= 0.

Insufficient identity, timing, units or source fields produce warnings and
missing values, not speculative ratios. With no fundamentals supplied, both BM
and log BM remain missing but those missing fields do not restrict the current illiquidity-only baseline.

## Optional daily factors

Use `--factors PATH`. The normalized file must have `date`, `RF`, `MKT_RF`,
`SMB`, `HML`; `Mkt-RF` is accepted after column normalization. Dates can be ISO
or YYYYMMDD. Declare `--factor-unit decimal|percent`; default decimal, never
auto-detected. No factors are downloaded. If an AVAILABLE_DATE column is supplied,
delayed releases cannot be used as contemporaneous factors and cause an error.

FF3 is the default; FF5 requires `--factor-model FF5` and RMW/CMA. The defaults
are `--idio-window 20 --idio-minimum 15`. Only the preceding stock observations
are used. Excess returns are residualized against an intercept and factors;
idio_vol is the sample residual standard deviation (ddof=1). Rank-deficient or
insufficient windows are missing. This internal factor-model fit is solely for
the requested volatility control, not a regression of ret_t1 on predictors.
It is an implementation choice, not a verified original-paper method.

## Validation

Stored source dates and stock trading lags are checked for every month: future
dependent date, lag-2/lag-21 return cutoffs, past-only illiquidity/idio cutoffs,
preceding-month ME, available BE/BM, historical network and SIC, and monthly
fixed controls. Any violated timing constraint raises an error. Network
self-loops, reversed/duplicate pairs, bad weights and node keys also raise.
Stock/date uniqueness, exact retained-row reconciliation and input mutation
checks run before success is recorded. Missing return windows remain missing;
no outlier exclusions or complete-case dropping occur.

The calculation currently holds the daily CRSP universe and stock controls in
memory, then writes one month at a time. Exact variable diagnostics reread one
numeric column at a time. Supplying daily factors adds per-stock rolling fits
and substantially increases runtime.

## Current baseline availability

The current Model 3 complete-case diagnostic uses ret_t1, reddit_ret, ret_d,
ret_20_2, ret_120_21 and illiquidity only. idio_vol, log_me, log_bm and ind_ret
remain in the panel for later specifications but do not restrict this baseline.
