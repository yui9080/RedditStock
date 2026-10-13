"""Minimal monthly-degree heterogeneity tests: two periods, Model 1/2 only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import runpy
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

GROUP = runpy.run_path(str(Path(__file__).with_name('05_02_degree_group_regression.py')))
BASE = GROUP['BASE']
ROOT = BASE['ROOT']
PERIODS = {'full_sample':('2014-06-01','2020-06-30'),
           'late_sample':('2019-01-01','2020-06-30')}


def linear_slope(params,covariance,contrast,df):
    slope=float(contrast@params)
    variance=float(contrast@covariance@contrast)
    if variance<=0 or not np.isfinite(variance):raise ValueError('Invalid contrast variance')
    se=float(np.sqrt(variance));t=slope/se
    critical=stats.t.ppf(.975,df)
    return {'slope':slope,'std_error':se,'t_stat':t,'p_value':float(2*stats.t.sf(abs(t),df)),
            'ci_low':float(slope-critical*se),'ci_high':float(slope+critical*se)}


def interaction_fit(sample,variables):
    x=sample[variables].copy();x.insert(0,'const',1.)
    for q in range(2,6):
        x[f'Q{q}']=sample.degree_group.eq(q).astype(float)
        x[f'reddit_ret_x_Q{q}']=sample.reddit_ret_pct*x[f'Q{q}']
    if any(x[c].nunique()<=1 for c in x if c!='const'):
        raise ValueError('Interaction specification has missing groups or zero-variance regressors')
    result=BASE['AbsorbingLS'](sample.ret_t1_pct,x,
        absorb=pd.DataFrame({'day':pd.Categorical(sample.date)},index=sample.index),drop_absorbed=False).fit(
        method='hdfe',cov_type='clustered',clusters=sample.permno,debiased=True,use_cache=False)
    factor=(result.nobs-x.shape[1])/(result.nobs-result.df_model)
    covariance=result.cov.to_numpy()*factor
    if not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T):raise ValueError('Invalid covariance matrix')
    return result,result.params.to_numpy(),covariance,list(result.params.index)


def contrasts(params,covariance,names,df):
    slopes=[]
    contrasts=[]
    for q in range(1,6):
        c=np.zeros(len(names));c[names.index('reddit_ret_pct')]=1
        if q>1:c[names.index(f'reddit_ret_x_Q{q}')]=1
        contrasts.append(c)
        slopes.append({'degree_quintile':f'Q{q}',**linear_slope(params,covariance,c,df)})
    restriction=np.zeros((4,len(names)))
    for i,q in enumerate(range(2,6)):restriction[i,names.index(f'reddit_ret_x_Q{q}')]=1
    difference=restriction@params
    restricted_cov=restriction@covariance@restriction.T
    if np.linalg.matrix_rank(restricted_cov)!=4:raise ValueError('Joint-test covariance is rank deficient')
    wald=float(difference@np.linalg.solve(restricted_cov,difference))
    fstat=wald/4
    q5=linear_slope(params,covariance,contrasts[4]-contrasts[0],df)
    tests=[{'test':'joint_heterogeneity','wald_stat':wald,'test_stat':fstat,'statistic_type':'F',
            'df_num':4,'df_den':df,'p_value':float(stats.f.sf(fstat,4,df))},
           {'test':'Q5_vs_Q1','slope_Q1':slopes[0]['slope'],'slope_Q5':slopes[4]['slope'],
            'difference_Q5_Q1':q5['slope'],'std_error_difference':q5['std_error'],
            'test_stat':q5['t_stat'],'statistic_type':'t','df_num':1,'df_den':df,'p_value':q5['p_value']}]
    return slopes,tests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    panel=args.panel or BASE['latest_panel']()
    panel_info=json.loads((panel.parent/'run_summary.json').read_text(encoding='utf-8'))
    network=Path(panel_info['network_run'])
    if not network.is_absolute():network=ROOT/network
    out=args.output_dir or ROOT/'Data/process/05_03_degree_heterogeneity_test'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    out.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();log=[];groups=[]
    input_stats={str(panel):(panel.stat().st_size,panel.stat().st_mtime_ns)}
    for file in sorted(network.glob('????-??/nodes.parquet')):
        input_stats[str(file)]=(file.stat().st_size,file.stat().st_mtime_ns)
        assigned=GROUP['degree_groups'](pd.read_parquet(file,columns=['permno','degree']),file.parent.name)
        if set(assigned.degree_group)!={1,2,3,4,5}:
            BASE['warning'](log,'Month cannot form all Q1-Q5; not silently ignored.',network_month=file.parent.name,groups=sorted(assigned.degree_group.unique().tolist()))
        groups.append(assigned)
    if not groups:raise FileNotFoundError('No monthly network nodes')
    groups=pd.concat(groups,ignore_index=True)
    if groups.duplicated(['network_month','permno']).any():raise ValueError('Duplicate monthly quintile identity')
    groups.to_parquet(out/'05_03_monthly_degree_quintiles.parquet',index=False,compression='zstd')
    frame,source_counts=GROUP['load_grouped_panel'](panel,groups,log)
    if not frame.degree.ge(0).all() or not frame.degree_group.isin([1,2,3,4,5]).all():raise ValueError('Invalid degree/quintile')
    summaries=[];all_slopes=[];all_tests=[];fits=[];text=[]
    for period,(start,end) in PERIODS.items():
        period_frame=frame.loc[frame.date.between(pd.Timestamp(start),pd.Timestamp(end))]
        for model in ['Model 1','Model 2']:
            variables=BASE['MODELS'][model]
            required=['ret_t1_pct']+variables
            sample=period_frame.loc[period_frame[required].notna().all(axis=1)].copy()
            if sample.empty:raise ValueError('Empty complete-case heterogeneity sample')
            if not np.isfinite(sample[required].to_numpy()).all():raise ValueError('Nonfinite regression values')
            for q in range(1,6):
                group_frame=period_frame.loc[period_frame.degree_group.eq(q)]
                rows,info,used=BASE['estimate'](group_frame,model,variables,log)
                main_row=next(r for r in rows if r['variable']=='reddit_ret_pct')
                dispersion=float(used.reddit_ret_pct.std(ddof=1)) if len(used)>1 else np.nan
                summaries.append({'sample':period,'model':model,'degree_quintile':f'Q{q}',
                    'N':len(used),'n_firms':info['n_firms'],'n_dates':info['n_dates'],
                    'min_date':info['min_date'],'max_date':info['max_date'],
                    'mean_degree':float(used.degree.mean()) if len(used) else np.nan,
                    'median_degree':float(used.degree.median()) if len(used) else np.nan,
                    'std_reddit_ret':dispersion,'beta_reddit_ret':main_row['coefficient'],
                    'std_error':main_row['std_error'],'t_stat':main_row['t_stat'],'p_value':main_row['p_value'],
                    'standardized_effect':main_row['coefficient']*dispersion,'status':info['status']})
            result,params,cov,names=interaction_fit(sample,variables)
            df=int(sample.permno.nunique())-1
            slopes,tests=contrasts(params,cov,names,df)
            for row in slopes:row.update(sample=period,model=model)
            for row in tests:row.update(sample=period,model=model)
            all_slopes.extend(slopes);all_tests.extend(tests)
            pd.DataFrame(cov,index=names,columns=names).to_csv(out/f'05_03_covariance_{period}_{model.replace(" ","_")}.csv')
            fits.append({'sample':period,'model':model,'formula':'ret_t1_pct ~ 1 + reddit_ret_pct + Q2-Q5 + reddit_ret_pct:Q2-Q5 + controls + day FE',
                'controls':variables[1:],**BASE['sample_details'](sample),
                'period_candidate_rows':len(period_frame),'dropped_complete_case':len(period_frame)-len(sample),
                'r_squared':float(result.rsquared),'adj_r_squared':float(result.rsquared_adj)})
            joint,comparison=tests
            text.append(f'\n{period} / {model}: '+json.dumps(fits[-1]))
            text.append('Although the subgroup coefficients differ numerically across degree quintiles, the formal joint test does not provide sufficient evidence of degree-related heterogeneity in the REDDIT RET slope.' if joint['p_value']>=.10 else
                        'The REDDIT RET slope differs significantly across degree quintiles, suggesting degree-related heterogeneity.')
            text.append('The Q5 coefficient is not statistically different from the Q1 coefficient.' if comparison['p_value']>=.10 else 'The REDDIT RET slope for Q5 is statistically different from Q1.')
            text.append(f'Joint F={joint["test_stat"]:.6g}, p={joint["p_value"]:.6g}; Q5-Q1={comparison["difference_Q5_Q1"]:.6g}, p={comparison["p_value"]:.6g}.')
            a,b=summaries[-5],summaries[-1]
            text.append(f'Q1 -> Q5 raw beta: {a["beta_reddit_ret"]:.6g} -> {b["beta_reddit_ret"]:.6g}; beta*SD(peer return): {a["standardized_effect"]:.6g} -> {b["standardized_effect"]:.6g} percentage points.')
            if b['beta_reddit_ret']>a['beta_reddit_ret']:
                text.append('The stronger high-degree coefficients are also accompanied by larger standardized economic effects.' if b['standardized_effect']>a['standardized_effect'] else
                            'The larger coefficients in high-degree groups may partly reflect lower dispersion of REDDIT RET rather than proportionally larger economic effects.')
            print(period,model,'joint p=',joint['p_value'],'Q5-Q1 p=',comparison['p_value'],flush=True)
    pd.DataFrame(summaries).to_csv(out/'05_03_degree_quintile_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_slopes).to_csv(out/'05_03_interaction_slopes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_tests).to_csv(out/'05_03_degree_heterogeneity_tests.csv',index=False,encoding='utf-8-sig')
    intro=('Subgroup estimates have separate day FE and control slopes; pooled interaction estimates share day FE and control slopes, and need not numerically match subgroup slopes.\n'
           'All returns are in percentage points. Standardized effect is beta times sample SD of reddit_ret_pct; no regression variables are standardized.\n'
           'Descriptive standardized-effect comparison uses Q1/Q5 endpoint directions only, not an additional significance test.\n'
           'Only joint heterogeneity and Q5-vs-Q1 are formally tested. Different subgroup significance alone does not establish different slopes. No causal inference.\n')
    (out/'05_03_degree_heterogeneity_summary.txt').write_text(intro+'\n'.join(text)+'\n',encoding='utf-8')
    for file,stat in input_stats.items():
        if stat!=(Path(file).stat().st_size,Path(file).stat().st_mtime_ns):raise ValueError('Input changed during test')
    report={'status':'complete','panel':str(panel),'panel_sha256':BASE['hash_file'](panel),'network_run':str(network),
            'periods':PERIODS,'pooled_fits':fits,'source_counts':source_counts,'warnings':log,
            'grouping':'05_02 monthly positive-degree nodes; ascending degree then PERMNO, deterministic tie splitting',
            'scaling':'returns multiplied by 100; standardized_effect uses SD on model/group complete cases, ddof=1',
            'covariance':'PERMNO CR1 including absorbed day FE degrees of freedom',
            'inference':'implied slopes and Q5-Q1 t(G-1); joint Wald/4 uses F(4,G-1)',
            'formal_tests':['joint_heterogeneity','Q5_vs_Q1'],'runtime_seconds':time.perf_counter()-tick,
            'output_directory':str(out)}
    (out/'run_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished:',out,flush=True)


if __name__=='__main__':main()
