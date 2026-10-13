"""Build a daily PERMNO panel. No outcome regression, winsorization or imputation.

python Notebook/05_build_regression_panel.py
Optional: --network-run PATH --crsp PATH --fundamentals PATH --factors PATH
See 05_build_regression_panel_README.md for input contracts and timing choices.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse


BASELINE = ['ret_t1', 'ret_d', 'reddit_ret', 'ret_20_2', 'ret_120_21',
            'illiquidity']
VARIABLES = BASELINE + ['idio_vol','log_me','log_bm','ind_ret','me', 'bm', 'siccd', 'n_reddit_peers', 'n_valid_peers_d',
                         'illiquidity_raw', 'illiquidity_scaled', 'me_daily', 'dlret']
UNITS = {'usd': 1.0, 'thousand_usd': 1000.0, 'million_usd': 1_000_000.0}
DATE_COLUMNS = ['date', 'ret_t1_date', 'lag2_date', 'lag21_date', 'lag1_date',
                'me_source_date', 'monthend_cutoff', 'bm_available_date',
                'bm_measurement_date', 'network_window_end', 'sic_source_date',
                'ret_20_2_max_date','ret_120_21_max_date','illiquidity_max_date','idio_vol_max_date']


def notify(log, message, **details):
    warnings.warn(message, stacklevel=2)
    log.append({'message': message, **details})


def read_table(path):
    return pd.read_parquet(path) if Path(path).suffix.lower() == '.parquet' else pd.read_csv(path)


def numeric(series):
    values = pd.to_numeric(series, errors='coerce')
    return values.where(np.isfinite(values))


def return_values(series):
    # Alphabetic CRSP codes and numeric missing sentinels (< -1) are not returns.
    values = numeric(series)
    return values.where(values.ge(-1))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def latest_network(root):
    runs = []
    for path in (root/'Data/process/04_monthly_ticker_network').glob('*/run_summary.json'):
        info = json.loads(path.read_text(encoding='utf-8'))
        if info.get('status') == 'complete' and info.get('node_unit') == 'permno':
            runs.append(path.parent)
    if not runs:
        raise FileNotFoundError('No completed PERMNO network; rerun 02_02 and 04 first.')
    return max(runs, key=lambda p: p.name)


def load_crsp(path, me_unit, log):
    aliases = {'permno': ['permno', 'PERMNO'], 'date': ['DlyCalDt', 'date', 'DATE'],
               'ret_d': ['DlyRet', 'RET', 'ret'], 'prc': ['DlyPrc', 'PRC', 'prc'],
               'vol': ['DlyVol', 'VOL', 'vol'], 'cap': ['DlyCap', 'ME', 'me'],
               'shrout': ['SHROUT', 'shrout'], 'dlret': ['DLRET', 'dlret'], 'siccd': ['siccd', 'SICCD']}
    header = pd.read_csv(path, nrows=0).columns
    chosen = {target: next((c for c in names if c in header), None) for target, names in aliases.items()}
    if any(chosen[c] is None for c in ['permno', 'date', 'ret_d']):
        raise ValueError('CRSP needs PERMNO, daily date and RET/DlyRet')
    if not chosen['prc'] or not chosen['vol']:
        notify(log,'Illiquidity lacks daily PRC/VOL and will be missing.',missing=[c for c in ['prc','vol'] if not chosen[c]])
    if chosen['prc'] and chosen['shrout']:
        me_method = 'abs(PRC)*SHROUT; SHROUT in thousands, converted to configured ME unit'
    elif chosen['cap']:
        me_method = f'{chosen["cap"]} as provided; configured unit={me_unit}'
        notify(log, 'No SHROUT: using supplied daily capitalization, not reconstructing it.', column=chosen['cap'], unit=me_unit)
    else:
        me_method = 'unavailable'
        notify(log, 'No PRC+SHROUT or capitalization: ME and log_me will be missing.')
    frames, rows, dropped, invalid_returns = [], 0, 0, 0
    columns = [c for c in chosen.values() if c]
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, chunksize=250_000):
        rows += len(chunk)
        if rows % 2_500_000 == 0: print(f'Read {rows:,} CRSP rows',flush=True)
        frame = pd.DataFrame(index=chunk.index)
        ids = numeric(chunk[chosen['permno']])
        dates = pd.to_datetime(chunk[chosen['date']], errors='coerce').dt.normalize()
        valid = ids.notna() & ids.gt(0) & ids.mod(1).eq(0) & dates.notna()
        dropped += int((~valid).sum())
        frame['permno'] = ids.loc[valid].astype('int64')
        frame['date'] = dates.loc[valid]
        frame = frame.loc[valid].copy()
        raw = chunk.loc[valid, chosen['ret_d']]
        frame['ret_d'] = return_values(raw)
        invalid_returns += int(frame.ret_d.isna().sum())
        for name in ['prc', 'vol', 'shrout', 'dlret', 'siccd']:
            frame[name] = (numeric(chunk.loc[valid, chosen[name]]) if chosen[name] else np.nan)
        if chosen['dlret']: frame['dlret'] = return_values(chunk.loc[valid, chosen['dlret']])
        if chosen['prc'] and chosen['shrout']:
            frame['me_daily'] = frame.prc.abs() * frame.shrout * 1000 / UNITS[me_unit]
        elif chosen['cap']:
            frame['me_daily'] = numeric(chunk.loc[valid, chosen['cap']])
        else:
            frame['me_daily'] = np.nan
        frame['me_daily'] = frame.me_daily.where(frame.me_daily.gt(0))
        frames.append(frame.drop(columns='shrout'))
    daily = pd.concat(frames, ignore_index=True).sort_values(['permno','date']).reset_index(drop=True)
    duplicate_mask = daily.duplicated(['permno','date'],keep=False)
    duplicate_dropped = 0
    if duplicate_mask.any():
        # Recheck every ORIGINAL source column, not just variables selected for
        # this panel. Different metadata or return strings cannot be merged.
        keys = daily.loc[duplicate_mask,['permno','date']].drop_duplicates()
        original_rows = []
        for chunk in pd.read_csv(path,dtype=str,keep_default_na=False,chunksize=250_000):
            chunk['__panel_permno'] = numeric(chunk[chosen['permno']])
            chunk['__panel_date'] = pd.to_datetime(chunk[chosen['date']],errors='coerce').dt.normalize()
            selected = chunk.merge(keys.rename(columns={'permno':'__panel_permno','date':'__panel_date'}),
                                   on=['__panel_permno','__panel_date'],how='inner',validate='many_to_one')
            if len(selected):original_rows.append(selected)
        originals = pd.concat(original_rows,ignore_index=True).drop_duplicates(subset=list(header))
        if originals.duplicated(['__panel_permno','__panel_date'],keep=False).any():
            raise ValueError('Conflicting PERMNO-date CRSP rows: source columns differ; resolve upstream.')
        duplicate_dropped = int(daily.duplicated(['permno','date']).sum())
        daily = daily.loc[~daily.duplicated(['permno','date'])].reset_index(drop=True)
        notify(log,'Dropped exactly identical CRSP source duplicates after verifying every original column.',
               count=duplicate_dropped,duplicate_keys=len(keys))
    if dropped: notify(log, 'Dropped CRSP rows with invalid PERMNO/date.', count=dropped)
    return daily, {'crsp_input_rows': rows, 'invalid_key_rows_dropped': dropped,
                   'exact_duplicate_rows_dropped': duplicate_dropped,
                   'crsp_observations_before_merge': len(daily), 'missing_or_invalid_ret_rows': invalid_returns,
                   'me_method': me_method, 'me_unit': me_unit, 'has_crsp_sic': bool(chosen['siccd'])}


def compounded_window(returns, shift, length):
    shifted = returns.shift(shift)
    valid = shifted.notna()
    zeros = shifted.eq(-1)
    # A -100% return makes the product zero. This avoids log(0) while preserving it.
    log_terms = np.log1p(shifted.where(~zeros, 0))
    summed = log_terms.rolling(length, min_periods=length).sum()
    zero_count = zeros.astype(float).rolling(length, min_periods=length).sum()
    count = valid.astype(float).rolling(length, min_periods=length).sum()
    with np.errstate(over='raise', invalid='raise'):
        result = np.expm1(summed)
    return result.mask(zero_count.gt(0), -1).where(count.eq(length))


def add_stock_controls(daily):
    group = daily.groupby('permno', sort=False)
    for name, shift in [('ret_t1_date', -1), ('lag1_date', 1), ('lag2_date', 2), ('lag21_date', 21)]:
        daily[name] = group.date.shift(shift)
    daily['ret_t1'] = group.ret_d.shift(-1)
    daily['me_weight_lag1'] = group.me_daily.shift(1)
    denominator = daily.prc.abs() * daily.vol
    daily['daily_illiq'] = daily.ret_d.abs().div(denominator.where(denominator.gt(0)))
    daily['ret_20_2'] = np.nan
    daily['ret_120_21'] = np.nan
    daily['illiquidity_raw'] = np.nan
    for index,positions in enumerate(group.indices.values(),1):
        returns = daily.loc[positions, 'ret_d']
        daily.loc[positions, 'ret_20_2'] = compounded_window(returns, 2, 19).to_numpy()
        daily.loc[positions, 'ret_120_21'] = compounded_window(returns, 21, 100).to_numpy()
        illiq = daily.loc[positions, 'daily_illiq'].shift(1).rolling(20, min_periods=10).mean()
        daily.loc[positions, 'illiquidity_raw'] = illiq.to_numpy()
        if index % 1000 == 0: print(f'Stock controls: {index:,} PERMNOs',flush=True)
    daily['illiquidity_scaled'] = daily.illiquidity_raw * 1e6
    daily['illiquidity'] = daily.illiquidity_scaled
    daily['ret_20_2_max_date'] = daily.lag2_date.where(daily.ret_20_2.notna())
    daily['ret_120_21_max_date'] = daily.lag21_date.where(daily.ret_120_21.notna())
    daily['illiquidity_max_date'] = daily.lag1_date.where(daily.illiquidity.notna())
    daily.drop(columns='daily_illiq', inplace=True)
    daily['network_month'] = daily.date.dt.to_period('M').astype(str)
    return daily


def add_sic(daily, path, has_crsp_sic, log):
    daily['sic_source_date'] = pd.NaT
    if has_crsp_sic:
        daily['sic_source_date'] = daily.date
    elif path and Path(path).is_file():
        history = read_table(path)
        required = {'permno','siccd','secinfostartdt','secinfoenddt'}
        if not required <= set(history):
            notify(log, 'SIC history lacks effective intervals; SIC and industry returns remain missing.', missing=sorted(required-set(history)))
            return
        history['permno'] = numeric(history.permno)
        history['siccd'] = numeric(history.siccd)
        history['start'] = pd.to_datetime(history.secinfostartdt, errors='coerce')
        history['end'] = pd.to_datetime(history.secinfoenddt, errors='coerce')
        usable = history.permno.gt(0) & history.permno.mod(1).eq(0) & history.siccd.between(1,9999) & history.siccd.mod(1).eq(0)
        usable &= history.start.notna() & history.end.notna() & history.start.le(history.end)
        records = {int(p): f for p,f in history.loc[usable].groupby('permno')}
        conflicts = 0
        for permno, positions in daily.groupby('permno', sort=False).indices.items():
            if permno not in records: continue
            dates = daily.loc[positions, 'date'].to_numpy()
            sic = np.full(len(positions), np.nan)
            starts = np.full(len(positions), np.datetime64('NaT'), dtype='datetime64[ns]')
            ambiguous = np.zeros(len(positions), dtype=bool)
            for row in records[permno].itertuples():
                active = (dates >= row.start.to_datetime64()) & (dates <= row.end.to_datetime64())
                ambiguous |= active & ~np.isnan(sic) & (sic != row.siccd)
                sic[active] = row.siccd
                old = starts[active]
                starts[active] = np.maximum(np.where(np.isnat(old), row.start.to_datetime64(), old), row.start.to_datetime64())
            sic[ambiguous] = np.nan; starts[ambiguous] = np.datetime64('NaT')
            conflicts += int(ambiguous.sum())
            daily.loc[positions, 'siccd'] = sic
            daily.loc[positions, 'sic_source_date'] = starts
        if conflicts: notify(log, 'Conflicting effective SIC codes left missing.', count=conflicts)
    else:
        notify(log, 'No point-in-time SIC history; SIC and industry returns remain missing.')
    valid = daily.siccd.between(1,9999) & daily.siccd.mod(1).eq(0)
    daily['siccd'] = daily.siccd.where(valid)


def monthly_me(daily):
    # Exact preceding calendar month, not arbitrary carry-forward across gaps.
    rows = daily.loc[daily.me_daily.gt(0), ['permno','date','network_month','me_daily']]
    ends = rows.groupby(['permno','network_month'], sort=False).tail(1).copy()
    ends['network_month'] = (pd.PeriodIndex(ends.network_month, freq='M') + 1).astype(str)
    ends.rename(columns={'date':'me_source_date', 'me_daily':'me'}, inplace=True)
    return ends


def prepare_fundamentals(path, me_unit, be_unit, log):
    if not path:
        notify(log, 'BM not built: no BM/Compustat input (requires PERMNO and actual availability dates).')
        return None
    data = read_table(path)
    if not {'permno','available_date'} <= set(data):
        notify(log, 'BM not built: missing PERMNO/available_date; datadate alone is not public availability.', missing=sorted({'permno','available_date'}-set(data)))
        return None
    data['permno'] = numeric(data.permno)
    data['available_date'] = pd.to_datetime(data.available_date, errors='coerce')
    measurement = next((c for c in ['measurement_date','datadate','date'] if c in data), None)
    if measurement is None:
        notify(log, 'BM not built: missing measurement_date/datadate/date; timing cannot be verified.')
        return None
    data['bm_measurement_date'] = pd.to_datetime(data[measurement], errors='coerce')
    valid = data.permno.gt(0) & data.permno.mod(1).eq(0) & data.available_date.notna() & data.bm_measurement_date.notna()
    if (~valid).any(): notify(log, 'Invalid fundamental identity/date records omitted.', count=int((~valid).sum()))
    data = data.loc[valid].copy()
    data['permno'] = data.permno.astype('int64')
    if data.available_date.lt(data.bm_measurement_date).any():
        raise ValueError('Fundamental availability precedes measurement date: possible look-ahead.')
    if data.duplicated(['permno','available_date']).any():
        raise ValueError('Duplicate PERMNO/available_date fundamentals; no silent version selection.')
    if 'bm' in data:
        data['bm_value'] = numeric(data.bm)
        data.attrs['mode'] = 'precomputed'
    else:
        if be_unit is None:
            notify(log, 'BM not built: book-equity unit must be explicitly supplied for Compustat inputs.')
            return None
        def col(name): return numeric(data[name]) if name in data else pd.Series(np.nan,index=data.index)
        preferred = col('pstkq')
        tax = col('txditcq').fillna(0)  # Optional tax term; explicit formula convention.
        be = col('seqq') - preferred
        # (ceqq+pstkq)-pstkq simplifies to ceqq without imputing preferred stock.
        be = be.where(col('seqq').notna(), col('ceqq'))
        be = be.where(col('seqq').notna() | col('ceqq').notna(), col('atq')-col('ltq')-preferred)
        data['be_in_me_units'] = (be+tax) * UNITS[be_unit]/UNITS[me_unit]
        data.attrs['mode'] = 'book_equity'
    return data


def add_bm(month, fundamentals):
    month['bm'] = np.nan
    month['bm_available_date'] = pd.NaT
    month['bm_measurement_date'] = pd.NaT
    if fundamentals is None or fundamentals.empty: return
    cutoff = month.monthend_cutoff.iloc[0]
    available = fundamentals.loc[fundamentals.available_date.le(cutoff)]
    latest = available.sort_values('available_date').groupby('permno',sort=False).tail(1).set_index('permno')
    if latest.empty: return
    month['bm_available_date'] = month.permno.map(latest.available_date)
    month['bm_measurement_date'] = month.permno.map(latest.bm_measurement_date)
    if fundamentals.attrs['mode'] == 'precomputed':
        # Do not label an old quarterly ratio as last-month-end BM.
        values = month.permno.map(latest.bm_value)
        prior_month = cutoff.to_period('M')
        valid = month.bm_measurement_date.dt.to_period('M').eq(prior_month)
        month['bm'] = values.where(valid)
    else:
        month['bm'] = month.permno.map(latest.be_in_me_units).div(month.me.where(month.me.gt(0)))


def load_factors(path, model, unit, log):
    names = ['MKT_RF','SMB','HML'] + (['RMW','CMA'] if model == 'FF5' else [])
    if not path:
        notify(log, 'Idiosyncratic volatility not built: no daily Fama-French factors; nothing downloaded.', missing=['date','RF']+names)
        return None
    factors = read_table(path)
    factors.rename(columns={c: c.strip().upper().replace('-','_') for c in factors}, inplace=True)
    factors.rename(columns={'DATE':'date'}, inplace=True)
    required = ['date','RF']+names
    if not set(required) <= set(factors):
        notify(log, 'Idiosyncratic volatility not built: factor columns missing.', missing=sorted(set(required)-set(factors)))
        return None
    dates = factors.date.astype(str)
    factors['date'] = pd.to_datetime(dates, format='mixed', errors='coerce')
    if factors.date.isna().any() or factors.date.duplicated().any(): raise ValueError('Invalid/duplicate factor dates')
    if 'AVAILABLE_DATE' in factors:
        available = pd.to_datetime(factors.AVAILABLE_DATE, errors='coerce')
        if available.isna().any() or available.gt(factors.date).any():
            raise ValueError('Factors include delayed/future availability; point-in-time merge required.')
    for col in ['RF']+names:
        factors[col] = numeric(factors[col]) / (100 if unit == 'percent' else 1)
    return factors.set_index('date')[['RF']+names]


def add_idio(daily, factors, model, window, minimum):
    daily['idio_vol'] = np.nan
    daily['idio_vol_max_date'] = pd.NaT
    if factors is None: return {'estimated_windows':0, 'rank_deficient_windows':0}
    aligned = factors.reindex(pd.DatetimeIndex(daily.date))
    columns = ['MKT_RF','SMB','HML'] + (['RMW','CMA'] if model == 'FF5' else [])
    y = daily.ret_d.to_numpy() - aligned.RF.to_numpy()
    x = np.column_stack([np.ones(len(daily)), aligned[columns].to_numpy()])
    estimated = deficient = 0
    result = np.full(len(daily),np.nan)
    for positions in daily.groupby('permno',sort=False).indices.values():
        for k in range(minimum,len(positions)):
            prior = positions[max(0,k-window):k]  # Exclusively before d.
            valid = np.isfinite(y[prior]) & np.isfinite(x[prior]).all(axis=1)
            if valid.sum() < minimum: continue
            chosen = prior[valid]
            coefficients, _, rank, _ = np.linalg.lstsq(x[chosen],y[chosen],rcond=None)
            if rank != x.shape[1]: deficient += 1; continue
            residual = y[chosen]-x[chosen]@coefficients
            result[positions[k]] = residual.std(ddof=1)
            estimated += 1
    daily['idio_vol'] = result
    daily['idio_vol_max_date'] = daily.lag1_date.where(daily.idio_vol.notna())
    return {'estimated_windows':estimated,'rank_deficient_windows':deficient}


def industry_returns(month):
    industry = np.floor(month.siccd/100)
    eligible = month.ret_d.notna() & month.me_weight_lag1.gt(0) & industry.notna()
    weights = month.me_weight_lag1.where(eligible,0)
    weighted = (month.ret_d * weights).where(eligible,0)
    groups = [month.date, industry]
    denominator = weights.groupby(groups).transform('sum')-weights
    numerator = weighted.groupby(groups).transform('sum')-weighted
    return numerator.div(denominator.where(denominator.gt(0))).where(industry.notna())


def validate_network(folder, timing):
    month = pd.Period(folder.name, freq='M')
    row = timing.loc[str(month)]
    if row['window_start'] != str(month-6) or row['window_end'] != str(month-1):
        raise ValueError(f'Network {month} does not use exactly prior six months.')
    edges = pd.read_parquet(folder/'edges.parquet')
    required = {'permno_i','permno_j','n_common_threads'}
    if not required <= set(edges): raise ValueError('Network must contain native PERMNO pairs, not ticker pairs.')
    for name in required:
        values = numeric(edges[name])
        if values.isna().any() or not values.gt(0).all() or not values.mod(1).eq(0).all():
            raise ValueError(f'Invalid network {name}')
        edges[name] = values.astype('int64')
    if not edges.permno_i.lt(edges.permno_j).all() or edges.duplicated(['permno_i','permno_j']).any():
        raise ValueError('Self-loop, noncanonical or duplicate PERMNO pair in network')
    nodes = pd.read_parquet(folder/'nodes.parquet').permno
    values = numeric(nodes)
    if values.isna().any() or not values.gt(0).all() or not values.mod(1).eq(0).all() or values.duplicated().any():
        raise ValueError('Invalid/duplicate network nodes')
    nodes = np.sort(values.to_numpy(dtype=np.int64))
    if not set(edges.permno_i).union(edges.permno_j) <= set(nodes): raise ValueError('Network edge endpoint absent from nodes')
    w = np.log1p(edges.n_common_threads.to_numpy(dtype=float))
    if 'weight' in edges and not np.allclose(edges.weight,w): raise ValueError('Stored network weight disagrees with log1p(raw weight)')
    i,j = np.searchsorted(nodes,edges.permno_i), np.searchsorted(nodes,edges.permno_j)
    rows, cols = np.concatenate([i,j]), np.concatenate([j,i])
    adjacency = sparse.csr_matrix((np.concatenate([w,w]),(rows,cols)),shape=(len(nodes),len(nodes)))
    binary = adjacency.copy(); binary.data[:] = 1
    return nodes, adjacency, binary, (month-1).end_time.normalize()


def reddit_returns(month, network):
    month['reddit_ret'] = np.nan
    month['n_reddit_peers'] = np.nan
    month['n_valid_peers_d'] = np.nan
    month['has_network_node'] = False
    month['network_window_end'] = pd.NaT
    if network is None: return
    nodes, adjacency, binary, cutoff = network
    degree = np.diff(adjacency.indptr)
    peers = pd.Series(degree,index=nodes)
    month['n_reddit_peers'] = month.permno.map(peers)
    month['has_network_node'] = month.permno.isin(nodes)
    month['network_window_end'] = cutoff
    for _, positions in month.groupby('date',sort=False).groups.items():
        today = month.loc[positions]
        indexed = today.set_index('permno').ret_d.reindex(nodes).to_numpy(dtype=float)
        valid = np.isfinite(indexed)
        # These temporary zero arrays are algebraic masks, not return imputations.
        numerator = adjacency @ np.where(valid,indexed,0)
        denominator = adjacency @ valid.astype(float)
        counts = binary @ valid.astype(float)
        averages = np.full(len(nodes),np.nan)
        np.divide(numerator,denominator,out=averages,where=denominator>0)
        month.loc[positions,'reddit_ret'] = today.permno.map(pd.Series(averages,index=nodes)).to_numpy()
        month.loc[positions,'n_valid_peers_d'] = today.permno.map(pd.Series(counts,index=nodes)).to_numpy()


def lookahead_checks(month):
    d = month.date
    if (month.ret_t1_date.notna() & month.ret_t1_date.le(d)).any(): raise ValueError('Nonfuture dependent-variable date')
    for value,limit in [('ret_20_2','lag2_date'),('ret_120_21','lag21_date'),
                        ('illiquidity','lag1_date'),('idio_vol','lag1_date')]:
        valid = month[value].notna()
        used = month[value+'_max_date']
        if (valid & (used.isna() | month[limit].isna() | used.gt(month[limit]) | used.ge(d))).any():
            raise ValueError(f'Invalid {value} source cutoff')
    prior_month = d.dt.to_period('M')-1
    if (month.me.notna() & (month.me_source_date.isna() | month.me_source_date.dt.to_period('M').ne(prior_month))).any():
        raise ValueError('ME does not come from exact preceding month')
    if (month.bm.notna() & (month.bm_available_date.isna() | month.bm_available_date.gt(month.monthend_cutoff)
                           | month.bm_measurement_date.gt(month.monthend_cutoff))).any():
        raise ValueError('BM contains unavailable/future information')
    if (month.network_window_end.notna() & month.network_window_end.ge(d.dt.to_period('M').dt.start_time)).any():
        raise ValueError('Contemporaneous/future network information')
    if (month.sic_source_date.notna() & month.sic_source_date.gt(d)).any(): raise ValueError('Future SIC information')
    for variable in ['me','bm','log_me','log_bm']:
        if month.groupby('permno')[variable].nunique(dropna=False).gt(1).any():
            raise ValueError(f'{variable} is not fixed within the network month')
    if month.duplicated(['permno','date']).any(): raise ValueError('Duplicate daily panel key')


def distribution(values):
    valid = values[np.isfinite(values)]
    n = len(valid)
    result = {'N':n,'missing_count':len(values)-n,'missing_pct':100*(len(values)-n)/len(values) if len(values) else 0}
    for name in ['mean','std','p1','p5','median','p95','p99','min','max']: result[name]=None
    if n:
        q=np.percentile(valid,[1,5,50,95,99],method='linear')
        result.update(mean=float(valid.mean()),std=float(valid.std(ddof=1)) if n>1 else None,
                      p1=float(q[0]),p5=float(q[1]),median=float(q[2]),p95=float(q[3]),p99=float(q[4]),
                      min=float(valid.min()),max=float(valid.max()))
    return result


def write_diagnostics(path, out, daily_info):
    parquet = pq.ParquetFile(path)
    summaries = []
    for variable in VARIABLES:
        pieces = [batch.column(0).to_numpy(zero_copy_only=False) for batch in parquet.iter_batches(columns=[variable])]
        values = np.concatenate(pieces).astype(float)
        summaries.append({'variable':variable,**distribution(values)})
    pd.DataFrame(summaries).to_csv(out/'04_variable_summary.csv',index=False,encoding='utf-8-sig')
    counts, year_counts, yearly_missing = {}, {}, {}
    counts.update({k:v for k,v in daily_info.items() if isinstance(v,int)})
    cols=list(dict.fromkeys(['date','has_network_node']+VARIABLES))
    for batch in parquet.iter_batches(columns=cols):
        f=batch.to_pandas(); year=f.date.dt.year
        measurements={'observations_with_valid_reddit_network':f.has_network_node,
                      'observations_with_reddit_ret':f.reddit_ret.notna(),
                      'observations_with_all_baseline_variables':f[BASELINE].notna().all(axis=1)}
        measurements.update({'observations_missing_'+c:f[c].isna() for c in ['ret_t1','illiquidity','idio_vol','log_me','log_bm']})
        for name,mask in measurements.items():counts[name]=counts.get(name,0)+int(mask.sum())
        for y,g in f.groupby(year):
            y=int(y);year_counts[y]=year_counts.get(y,0)+len(g)
            for variable in VARIABLES:
                key=(y,variable);yearly_missing[key]=yearly_missing.get(key,0)+int(g[variable].isna().sum())
    pd.DataFrame([{'metric':k,'count':v} for k,v in counts.items()]).to_csv(out/'04_merge_diagnostics.csv',index=False,encoding='utf-8-sig')
    rows=[{'year':y,'variable':v,'observation_count':year_counts[y],'missing_count':count,
           'missing_pct':100*count/year_counts[y]} for (y,v),count in sorted(yearly_missing.items())]
    pd.DataFrame(rows).to_csv(out/'04_yearly_missing_rates.csv',index=False,encoding='utf-8-sig')
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network-run',type=Path)
    parser.add_argument('--crsp',type=Path)
    parser.add_argument('--sic-history',type=Path)
    parser.add_argument('--fundamentals',type=Path)
    parser.add_argument('--factors',type=Path)
    parser.add_argument('--me-unit',choices=UNITS,default='thousand_usd')
    parser.add_argument('--book-equity-unit',choices=UNITS)
    parser.add_argument('--factor-model',choices=['FF3','FF5'],default='FF3')
    parser.add_argument('--factor-unit',choices=['decimal','percent'],default='decimal')
    parser.add_argument('--idio-window',type=int,default=20)
    parser.add_argument('--idio-minimum',type=int,default=15)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if not 2 <= args.idio_minimum <= args.idio_window:parser.error('Require 2 <= minimum <= window')
    root=Path(__file__).resolve().parents[1]
    network_run=args.network_run or latest_network(root)
    crsp=args.crsp or root/'Data/raw/stock/RAW_crsp.csv'
    sic=args.sic_history or root/'Data/raw/stock/RAW_name.csv'
    manifest=json.loads((network_run/'run_summary.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or manifest.get('node_unit')!='permno':raise ValueError('Use completed native PERMNO network')
    timing=pd.read_csv(network_run/'monthly_network_summary.csv',dtype={'network_month':str,'window_start':str,'window_end':str}).set_index('network_month')
    if timing.index.duplicated().any():raise ValueError('Duplicate network month')
    folders={p.parent.name:p.parent for p in network_run.glob('????-??/edges.parquet')}
    if set(folders)!=set(timing.index):raise ValueError('Network files differ from monthly manifest')
    out=args.output_dir or root/'Data/process/05_regression_panel'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();log=[]
    inputs=[crsp,network_run/'run_summary.json',network_run/'monthly_network_summary.csv']
    inputs += [folder/name for folder in folders.values() for name in ['edges.parquet','nodes.parquet']]
    inputs += [p for p in [sic,args.fundamentals,args.factors] if p and p.is_file()]
    input_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
    print('Network:',network_run,'\nOutput:',out,flush=True)
    daily,info=load_crsp(crsp,args.me_unit,log)
    print(f'Loaded {len(daily):,} CRSP observations; building lagged stock controls',flush=True)
    daily=add_stock_controls(daily)
    add_sic(daily,sic,info['has_crsp_sic'],log)
    ends=monthly_me(daily)
    fundamentals=prepare_fundamentals(args.fundamentals,args.me_unit,args.book_equity_unit,log)
    factors=load_factors(args.factors,args.factor_model,args.factor_unit,log)
    idio_info=add_idio(daily,factors,args.factor_model,args.idio_window,args.idio_minimum)
    if (daily.me_weight_lag1.notna() & (daily.lag1_date.isna() | daily.lag1_date.ge(daily.date))).any():
        raise ValueError('Industry ME weight is not lagged')
    temp=out/'04_regression_panel.parquet.partial';dest=out/'04_regression_panel.parquet'
    writer=None;output_rows=0;month_info=[]
    try:
        for index,(month,positions) in enumerate(daily.groupby('network_month',sort=True).indices.items(),1):
            tick=time.perf_counter()
            frame=daily.iloc[positions].copy()
            frame=frame.merge(ends,on=['permno','network_month'],how='left',validate='many_to_one')
            frame['me_lag_monthend']=frame.me
            frame['monthend_cutoff']=(pd.Period(month,freq='M')-1).end_time.normalize()
            frame['log_me']=np.log(frame.me.where(frame.me.gt(0)))
            add_bm(frame,fundamentals)
            frame['log_bm']=np.log(frame.bm.where(frame.bm.gt(0)))
            frame['ind_ret']=industry_returns(frame)
            network=validate_network(folders[month],timing) if month in folders else None
            reddit_returns(frame,network)
            lookahead_checks(frame)
            for col in DATE_COLUMNS:frame[col]=pd.to_datetime(frame[col]).astype('datetime64[ns]')
            # Explicit types keep all-missing early months compatible with later
            # populated months (e.g. a degree column cannot switch float -> int).
            panel_schema=pa.schema([
                (col, pa.timestamp('ns') if col in DATE_COLUMNS else
                 pa.int64() if col=='permno' else pa.string() if col=='network_month' else
                 pa.bool_() if col=='has_network_node' else pa.float64()) for col in frame.columns
            ])
            table=pa.Table.from_pandas(frame,schema=panel_schema,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(temp,panel_schema,compression='zstd')
            writer.write_table(table,row_group_size=100_000)
            output_rows+=len(frame)
            month_info.append({'network_month':month,'rows':len(frame),'network_available':network is not None,
                               'network_nodes_with_crsp':int(frame.has_network_node.sum()),
                               'rows_with_reddit_ret':int(frame.reddit_ret.notna().sum()),'seconds':time.perf_counter()-tick})
            if index==1 or index%12==0:print(f'{month}: {len(frame):,} rows, Reddit return available on {int(frame.reddit_ret.notna().sum()):,}',flush=True)
    finally:
        if writer:writer.close()
    if output_rows!=len(daily) or pq.read_metadata(temp).num_rows!=output_rows:raise ValueError('Panel row reconciliation failed')
    for path,(size,modified) in input_stats.items():
        if (Path(path).stat().st_size,Path(path).stat().st_mtime_ns)!=(size,modified):raise ValueError('Input changed during run')
    temp.replace(dest)
    diagnostics=write_diagnostics(dest,out,info)
    absent=[r['network_month'] for r in month_info if not r['network_available']]
    if absent:notify(log,'CRSP months without a corresponding network retained with missing peer returns.',months=absent)
    report={'status':'complete_with_warnings' if log else 'complete','output':str(dest),'output_rows':output_rows,
            'network_run':str(network_run),'crsp_source':str(crsp),'crsp_sha256':file_hash(crsp),
            'input_stats':input_stats,'diagnostics':diagnostics,'months':month_info,'warnings':log,
            'me_unit':args.me_unit,'book_equity_unit':args.book_equity_unit,'me_method':info['me_method'],
            'idio_model':args.factor_model,'idio_window':args.idio_window,'idio_minimum':args.idio_minimum,
            'factor_unit':args.factor_unit,'idio_built':factors is not None,'idio_diagnostics':idio_info,
            'bm_built':fundamentals is not None,'regression_executed':False,
            'illiquidity_note':'Amihud illiquidity is an implementation choice because Wang and Zhang (2024) do not provide an explicit formula in the available paper text.',
            'idio_note':'FF3/previous-20-day residual standard deviation (ddof=1) is an implementation choice, not a verified original-paper specification.',
            'illiquidity_scale':1e6,'illiquidity_window':'previous 20 stock trading observations, minimum 10 valid',
            'return_windows':{'ret_20_2':'shift(2), 19 observations','ret_120_21':'shift(21), 100 observations'},
            'industry_weight':'previous stock trading observation ME; all CRSP stocks, same effective two-digit SIC, focal excluded',
            'baseline_variables':BASELINE, 'missing_values_imputed':False,'winsorized':False,'lookahead_checks':'passed for every output month',
            'runtime_seconds':time.perf_counter()-started}
    (out/'run_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished:',dest,flush=True)


if __name__=='__main__':main()
