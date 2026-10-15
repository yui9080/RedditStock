"""Unweighted baseline daily regressions, day FE and PERMNO-clustered SE.

python Notebook/05_01_baseline_regression.py [--sample both|full_sample|late_sample] [--panel PATH]
Default runs both the 2012-11/2020-08 full sample and 2019-01/2020-08 late sample. Model 3 is reported as
not estimable when its full complete-case sample is empty; controls are never
silently removed. Return coefficients use actual percent/percent units.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_samples import SAMPLE_PERIODS as PERIODS, SAMPLE_ALIASES
LOCAL_DEPS = ROOT/'.tmp_regression_deps'
if LOCAL_DEPS.is_dir(): sys.path.insert(0,str(LOCAL_DEPS))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
try:
    from linearmodels.iv.absorbing import AbsorbingLS
except ImportError as exc:
    raise ImportError('Install linearmodels and its dependencies; see 05_01_baseline_regression_README.md') from exc


RETURNS = ['ret_t1','reddit_ret','ret_d','ret_20_2','ret_120_21']
MODELS = {
    'Model 1':['reddit_ret_pct'],
    'Model 2':['reddit_ret_pct','ret_d_pct','ret_20_2_pct','ret_120_21_pct'],
    'Model 3':['reddit_ret_pct','ret_d_pct','ret_20_2_pct','ret_120_21_pct','illiquidity'],
}
LABELS = {'reddit_ret_pct':'REDDIT RET','ret_d_pct':'RET d','ret_20_2_pct':'RET [d-20,d-2]',
          'ret_120_21_pct':'RET [d-120,d-21]','illiquidity':'Illiquidity'}
RESULT_COLUMNS=['model','variable','coefficient','std_error','t_stat','p_value','n_obs','n_firms',
                'r_squared','adj_r_squared','coefficient_times_100','status']


def warning(log,message,**details):
    warnings.warn(message,stacklevel=2)
    log.append({'message':message,**details})


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def latest_panel():
    runs=[]
    for p in (ROOT/'Data/process/05_regression_panel').glob('*/run_summary.json'):
        info=json.loads(p.read_text(encoding='utf-8'))
        file=p.parent/'04_regression_panel.parquet'
        if info.get('status') in ['complete','complete_with_warnings'] and file.is_file():runs.append(file)
    if not runs:raise FileNotFoundError('No completed regression panel')
    return max(runs,key=lambda p:p.parent.name)


def load_sample(panel,start,end,log,network_years=None):
    required=['permno','date']+RETURNS+['illiquidity']
    if network_years is not None: required += ['network_month','network_window_end']
    if not set(required)<=set(pq.read_schema(panel).names):raise ValueError('Missing baseline panel columns')
    pieces=[];all_rows=invalid_dates=period_rows=0
    for batch in pq.ParquetFile(panel).iter_batches(columns=required,batch_size=200_000):
        frame=batch.to_pandas();all_rows+=len(frame)
        dates=pd.to_datetime(frame.date,errors='coerce')
        invalid_dates+=int(dates.isna().sum())
        frame['date']=dates
        keep=dates.between(start,end)
        period_rows+=int(keep.sum())
        if network_years is not None:
            months=pd.to_datetime(frame.network_month,format='%Y-%m',errors='coerce')
            cutoff=pd.to_datetime(frame.network_window_end,errors='coerce')
            if (cutoff.notna() & cutoff.ge(months)).any(): raise ValueError('Future/contemporaneous network cutoff')
            keep &= months.dt.year.isin(network_years) & cutoff.notna()
        pieces.append(frame.loc[keep])
    frame=pd.concat(pieces,ignore_index=True)
    if invalid_dates:raise ValueError('Panel contains invalid dates; rebuild the panel')
    if frame.empty:raise ValueError('No observations in requested period')
    if frame.duplicated(['permno','date']).any():raise ValueError('Duplicate PERMNO-date; cannot continue regression')
    ids=pd.to_numeric(frame.permno,errors='coerce')
    if ids.isna().any() or not ids.gt(0).all() or not ids.mod(1).eq(0).all():raise ValueError('Invalid PERMNO')
    frame['permno']=ids.astype('int64')
    for variable in RETURNS+['illiquidity']:
        values=pd.to_numeric(frame[variable],errors='coerce')
        invalid=~np.isfinite(values)
        if invalid.any():warning(log,f'{variable}: missing/nonfinite values; each model applies its own complete-case filter.',count=int(invalid.sum()))
        frame[variable]=values.where(np.isfinite(values))
    for variable in RETURNS:frame[variable+'_pct']=frame[variable]*100
    frame.attrs['period_rows_before_network_filter']=period_rows
    frame.attrs['excluded_network_filter']=period_rows-len(frame)
    return frame,all_rows


def sample_details(frame):
    return {'n_obs':len(frame),'n_firms':int(frame.permno.nunique()),'n_dates':int(frame.date.nunique()),
            'min_date':str(frame.date.min().date()) if len(frame) else None,
            'max_date':str(frame.date.max().date()) if len(frame) else None}


def estimate(frame,model,variables,log):
    tick=time.perf_counter()
    needed=['ret_t1_pct']+variables
    missing=frame[needed].isna().sum().to_dict()
    sample=frame.loc[frame[needed].notna().all(axis=1)].copy()
    info={'model':model,'formula':'ret_t1_pct ~ 1 + '+' + '.join(variables)+' + day FE',
          **sample_details(sample),'candidate_observations':len(frame),
          'dropped_missing':len(frame)-len(sample),'missing_by_variable':{k:int(v) for k,v in missing.items()}}
    if sample.empty or info['n_firms']<2:
        info.update(status='not_estimable',reason='No complete cases' if sample.empty else 'Fewer than two PERMNO clusters')
        warning(log,f'{model} not estimable; requested specification retained.',**{'reason':info['reason'],'missing_by_variable':info['missing_by_variable']})
        rows=[dict(model=model,variable=v,coefficient=np.nan,std_error=np.nan,t_stat=np.nan,p_value=np.nan,
                   n_obs=len(sample),n_firms=info['n_firms'],r_squared=np.nan,adj_r_squared=np.nan,
                   coefficient_times_100=np.nan,status='not_estimable') for v in variables]
        return rows,info,sample
    zero=[v for v in variables if sample[v].nunique()<=1]
    if zero:
        warning(log,f'{model}: zero variance; cannot estimate requested regressors.',variables=zero)
        raise ValueError('Zero-variance regressors: '+str(zero))
    x=sample[variables].copy();x.insert(0,'const',1.0)
    absorb=pd.DataFrame({'day':pd.Categorical(sample.date)},index=sample.index)
    # The library absorbs FE internally. No manual demeaning or standardization.
    regression=AbsorbingLS(sample.ret_t1_pct,x,absorb=absorb,drop_absorbed=False)
    result=regression.fit(method='hdfe',cov_type='clustered',clusters=sample.permno,
                          debiased=True,use_cache=False)
    rows=[]
    cluster_df=info['n_firms']-1
    # AbsorbingLS's clustered correction counts explicit regressors, whereas
    # ordinary dummy-variable OLS counts the absorbed day effects as well.
    # Include all FE degrees of freedom to match the full-design CR1 correction.
    fe_df_correction=np.sqrt((result.nobs-x.shape[1])/(result.nobs-result.df_model))
    for variable in result.params.index:
        coefficient=float(result.params[variable]);se=float(result.std_errors[variable])*fe_df_correction
        t=coefficient/se;p=float(2*stats.t.sf(abs(t),df=cluster_df))
        rows.append(dict(model=model,variable=variable,coefficient=coefficient,std_error=se,t_stat=t,p_value=p,
                         n_obs=int(result.nobs),n_firms=info['n_firms'],r_squared=float(result.rsquared),
                         adj_r_squared=float(result.rsquared_adj),coefficient_times_100=100*coefficient,status='estimated'))
    main=next(row for row in rows if row['variable']=='reddit_ret_pct')
    info.update(status='estimated',r_squared=float(result.rsquared),adj_r_squared=float(result.rsquared_adj),
                df_model=int(result.df_model),p_value_reference_df=cluster_df,
                absorbed_fe_standard_error_multiplier=float(fe_df_correction),
                reddit_coefficient=main['coefficient'],reddit_t_stat=main['t_stat'],reddit_p_value=main['p_value'],
                coefficient_sign='positive' if main['coefficient']>0 else 'negative' if main['coefficient']<0 else 'zero',
                significant_5pct=main['p_value']<.05,positive_sign_consistent_with_user_paper_reference=main['coefficient']>0,
                runtime_seconds=time.perf_counter()-tick)
    return rows,info,sample


def diagnostics(sample,out,log,prefix='05_01'):
    variables=['ret_t1_pct']+MODELS['Model 3']
    counts=sample.groupby('permno').size()
    rows=[{'kind':'sample','variable':'Model 3','n_obs':len(sample),'n_firms':sample.permno.nunique(),
           'n_dates':sample.date.nunique(),'mean_observations_per_firm':counts.mean() if len(counts) else np.nan,
           'median_observations_per_firm':counts.median() if len(counts) else np.nan}]
    for variable in variables:
        values=sample[variable]
        stats_row={'kind':'variable','variable':variable,'N':len(values),
                   **dict.fromkeys(['mean','std','min','p1','p5','median','p95','p99','max'],np.nan)}
        if len(values):
            q=values.quantile([.01,.05,.5,.95,.99])
            stats_row.update(mean=values.mean(),std=values.std(),min=values.min(),p1=q.loc[.01],p5=q.loc[.05],
                             median=q.loc[.5],p95=q.loc[.95],p99=q.loc[.99],max=values.max())
        rows.append(stats_row)
    pd.DataFrame(rows).to_csv(out/f'{prefix}_regression_diagnostics.csv',index=False,encoding='utf-8-sig')
    correlation=sample[MODELS['Model 3']].corr()
    correlation.to_csv(out/f'{prefix}_pairwise_correlations.csv',encoding='utf-8-sig')
    if sample.empty:warning(log,'Model 3 diagnostics and correlations unavailable: empty complete-case sample.')
    else:
        for i,a in enumerate(correlation.columns):
            for b in correlation.columns[i+1:]:
                if abs(correlation.loc[a,b])>.9:warning(log,'High pairwise correlation in Model 3.',variables=[a,b],correlation=float(correlation.loc[a,b]))
        if not np.isfinite(sample[variables].to_numpy()).all():warning(log,'Nonfinite values in Model 3 diagnostic sample')


def stars(p):return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''


def write_table(results,infos,out,prefix='05_01'):
    rows=[]
    for variable,label in LABELS.items():
        coefficient={'variable':label};t_stat={'variable':''}
        for number,(model,requested) in enumerate(MODELS.items(),1):
            cell=results.loc[(results.model==model)&results.variable.eq(variable)]
            if len(cell) and cell.status.iloc[0]=='estimated':
                r=cell.iloc[0];coefficient[f'({number})']=f'{r.coefficient:.6f}{stars(r.p_value)}'
                t_stat[f'({number})']=f'({r.t_stat:.2f})'
            else:coefficient[f'({number})']='NA' if variable in requested else '';t_stat[f'({number})']=''
        rows.extend([coefficient,t_stat])
    for label,value in [('Day FE','Yes'),('Firm FE','No'),('Firm-clustered SE','Yes')]:
        rows.append({'variable':label,**{f'({i})':value for i in range(1,4)}})
    for label,key in [('Observations','n_obs'),('Firms','n_firms'),('Trading days','n_dates'),('Adj. R2','adj_r_squared'),('Status','status')]:
        rows.append({'variable':label,**{f'({i})':infos[i-1].get(key,'NA') for i in range(1,4)}})
    pd.DataFrame(rows).to_csv(out/f'{prefix}_baseline_table.csv',index=False,encoding='utf-8-sig')


def interpretation(info):
    note=('Scaling: both dependent and return explanatory variables are in percentage points. '
          'Actual slope b implies b percentage points of next-day focal return per 1 percentage point of peer return. '
          'For a percent-of-movement report, beta_report=100*b and carryover_ratio=beta_report/100=b. '
          'Dividing the actual percent/percent slope by 100 again is incorrect. This is association, not causation.\n')
    if info['status']!='estimated':return 'Model 3 not estimated: '+info['reason']+'. No numerical interpretation available.\n'+note
    b=info['reddit_coefficient']
    return (f'Model 3 actual slope: {b:.8g}. Reported percent-of-movement slope: {100*b:.8g}.\n'
            f'A 1 percentage-point increase in Reddit peer return on day d is associated with '
            f'a {b:.8g} percentage-point change in the focal return on its next trading observation.\n'
            f'carryover_ratio = beta_report / 100 = {b:.8g}, equivalent to {100*b:.8g}% of the peer-return movement.\n'+note)


SAMPLE_CONFIGS = {
    'full_sample': {'start':PERIODS['full_sample'][0],'end':PERIODS['full_sample'][1],'network_years':None},
    'late_sample': {'start':PERIODS['late_sample'][0],'end':PERIODS['late_sample'][1],'network_years':(2019,2020)},
}


def run_regression(panel,start,end,prefix,network_years=None,output_dir=None):
    source_stat=(panel.stat().st_size,panel.stat().st_mtime_ns)
    out=output_dir or ROOT/f'Data/process/{prefix}_baseline_regression'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    out.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();log=[]
    frame,total=load_sample(panel,start,end,log,network_years=network_years)
    info_all=sample_details(frame)
    print('Panel:',panel,'\nOutput:',out,'\nPeriod sample:',info_all,flush=True)
    rows=[];infos=[]
    for model,variables in MODELS.items():
        print('Estimating',model,flush=True)
        model_rows,info,sample=estimate(frame,model,variables,log)
        rows.extend(model_rows);infos.append(info)
        print(info,flush=True)
        if model=='Model 3':diagnostics(sample,out,log,prefix)
        del sample
    results=pd.DataFrame(rows,columns=RESULT_COLUMNS)
    results.to_csv(out/f'{prefix}_baseline_results.csv',index=False,encoding='utf-8-sig')
    write_table(results,infos,out,prefix)
    lines=[f'Requested sample: {start.strftime('%Y-%m-%d')} <= date <= {end.strftime('%Y-%m-%d')}',f'Period-restricted panel: {info_all}',
           f'Panel rows outside requested date period: {total-frame.attrs["period_rows_before_network_filter"]}',
           f'Network-year filter: {network_years}; rows excluded: {frame.attrs["excluded_network_filter"]}',
           'Day FE: Yes. Firm FE: No. Unweighted OLS. Covariance: one-way PERMNO cluster, CR1 including absorbed FE degrees of freedom.',
           'p-values: Student t with number of PERMNO clusters minus 1 degrees of freedom.',
           'Model 3 = Model 2 + illiquidity; idio_vol, log_me, BM/log_bm are excluded.',
           'Return variables multiplied by 100; other variables retain their panel scales.',
           'No imputation, winsorization, trimming, standardization or manual demeaning.',
           'Date restriction applies to predictor date d; ret_t1 may refer to the next trading observation after the period end.',
           'Paper comparison uses the user-supplied positive/significant reference only; different magnitudes are not a software error.']
    for info in infos:
        lines.extend(['',info['model']+': '+info['formula'],json.dumps(info,ensure_ascii=False)])
    lines.extend(['','Warnings:']+[json.dumps(w,ensure_ascii=False) for w in log])
    (out/f'{prefix}_baseline_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/f'{prefix}_interpretation.txt').write_text(interpretation(infos[2]),encoding='utf-8')
    if source_stat!=(panel.stat().st_size,panel.stat().st_mtime_ns):raise ValueError('Panel changed during regression')
    metadata={'status':'complete_with_unestimated_models' if any(i['status']!='estimated' for i in infos) else 'complete',
              'stage':prefix,'panel':str(panel),'panel_sha256':hash_file(panel),'requested_start':start.strftime('%Y-%m-%d'),'requested_end':end.strftime('%Y-%m-%d'),
              'network_years':list(network_years) if network_years is not None else None,
              'network_months':sorted(frame.network_month.unique().tolist()) if network_years is not None else None,
              'period_rows_before_network_filter':frame.attrs['period_rows_before_network_filter'],
              'excluded_network_filter':frame.attrs['excluded_network_filter'],
              'period_sample':info_all,'models':infos,'warnings':log,'day_fixed_effects':True,'firm_fixed_effects':False,
              'cluster':'PERMNO','covariance_debiased':True,'p_value_df':'n_firms-1',
              'covariance_correction':'CR1: G/(G-1) * (N-1)/(N-K_total), including absorbed day FE',
              'coefficient_scale':'percent-point dependent / percent-point return predictors; coefficient is actual slope',
              'paper_reporting_scale':'coefficient_times_100; not used in estimation',
              'carryover_ratio_definition':'coefficient_times_100 / 100 = actual slope',
              'versions':{p:importlib.metadata.version(p) for p in ['linearmodels','pandas','numpy','scipy']},
              'runtime_seconds':time.perf_counter()-tick,'output_directory':str(out)}
    (out/'run_summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished:',out,flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',choices=['both',*SAMPLE_CONFIGS,*SAMPLE_ALIASES],default='both')
    parser.add_argument('--panel',type=Path)
    parser.add_argument('--start',help='Optional date override for selected sample(s)')
    parser.add_argument('--end',help='Optional date override for selected sample(s)')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    panel=args.panel or latest_panel()
    selected=list(SAMPLE_CONFIGS) if args.sample=='both' else [SAMPLE_ALIASES.get(args.sample,args.sample)]
    plans=[]
    for name in selected:
        config=SAMPLE_CONFIGS[name]
        start=pd.Timestamp(args.start or config['start'])
        end=pd.Timestamp(args.end or config['end'])
        if start>end:parser.error('start must not exceed end')
        plans.append((name,config,start,end))
    run_root=args.output_dir or ROOT/'Data/process/05_01_baseline_regression'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    run_root.mkdir(parents=True,exist_ok=False)
    outputs={}
    for name,config,start,end in plans:
        print('Sample:',name,flush=True)
        output=run_regression(panel,start,end,'05_01',config['network_years'],run_root/name)
        outputs[name]=str(output)
    manifest={'sample_periods':PERIODS,'stage':'05_01','status':'complete','panel':str(panel),'samples':outputs,
              'output_directory':str(run_root)}
    (run_root/'run_summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Combined output:',run_root,flush=True)


if __name__=='__main__':main()
