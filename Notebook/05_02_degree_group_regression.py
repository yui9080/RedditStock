"""Monthly degree quintile regressions, full network period and 2019/2020.

Uses exactly the Model 1/2/3 estimation implementation of 05_01. Quintiles use
positive-degree network nodes, sorted by (degree, PERMNO) within each month.
Equal-degree ties may cross groups; no future returns enter the assignment.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import runpy
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = runpy.run_path(str(Path(__file__).with_name('05_01_baseline_regression.py')))
ROOT = BASE['ROOT']


def degree_groups(nodes, month):
    if nodes.permno.duplicated().any(): raise ValueError('Duplicate network PERMNO')
    for col in ['permno','degree']:
        ids=pd.to_numeric(nodes[col],errors='coerce')
        if ids.isna().any() or not np.isfinite(ids).all() or not ids.mod(1).eq(0).all():raise ValueError('Invalid network degree/PERMNO')
        nodes[col]=ids.astype('int64')
    if not nodes.permno.gt(0).all() or not nodes.degree.ge(0).all():raise ValueError('Invalid network nodes')
    linked=nodes.loc[nodes.degree.gt(0)].sort_values(['degree','permno']).copy()
    linked['degree_group']=(np.arange(len(linked))*5//len(linked)+1) if len(linked) else pd.Series(dtype='int64')
    linked['network_month']=month
    return linked[['network_month','permno','degree','degree_group']]


def load_grouped_panel(panel,groups,log):
    columns=['permno','date','network_month','network_window_end','n_reddit_peers','has_network_node','illiquidity']+BASE['RETURNS']
    pieces=[];total=excluded_no_link=0
    for batch in pq.ParquetFile(panel).iter_batches(columns=columns,batch_size=200_000):
        frame=batch.to_pandas();total+=len(frame)
        dates=pd.to_datetime(frame.date,errors='coerce')
        if dates.isna().any():raise ValueError('Invalid trading dates')
        frame['date']=dates
        months=pd.to_datetime(frame.network_month,format='%Y-%m',errors='coerce')
        if not months.dt.to_period('M').eq(dates.dt.to_period('M')).all():raise ValueError('Panel month mismatch')
        cutoff=pd.to_datetime(frame.network_window_end,errors='coerce')
        if (cutoff.notna() & cutoff.ge(months)).any():raise ValueError('Future network information')
        keep=frame.has_network_node & frame.n_reddit_peers.gt(0) & cutoff.notna()
        excluded_no_link+=int((~keep).sum())
        pieces.append(frame.loc[keep])
    frame=pd.concat(pieces,ignore_index=True)
    if frame.duplicated(['permno','date']).any():raise ValueError('Duplicate PERMNO-date')
    frame=frame.merge(groups,on=['permno','network_month'],how='left',validate='many_to_one')
    if frame.degree_group.isna().any():raise ValueError('Positive-degree panel stock missing grouping assignment')
    if not frame.degree.eq(frame.n_reddit_peers).all():raise ValueError('Network degree differs from panel peer count')
    for col in BASE['RETURNS']+['illiquidity']:
        values=pd.to_numeric(frame[col],errors='coerce')
        invalid=~np.isfinite(values)
        if invalid.any():BASE['warning'](log,f'{col}: missing/nonfinite; model-specific complete cases.',count=int(invalid.sum()))
        frame[col+'_pct' if col in BASE['RETURNS'] else col]=values.where(np.isfinite(values))*(100 if col in BASE['RETURNS'] else 1)
    return frame,{'panel_rows':total,'excluded_without_positive_degree_network':excluded_no_link,
                   'positive_degree_candidate_rows':len(frame)}


def paper_table(results,infos,model,out):
    variables=BASE['MODELS'][model]
    rows=[]
    for variable in variables:
        coefficient={'variable':BASE['LABELS'][variable]};tstat={'variable':''}
        for group in range(1,6):
            selected=results.loc[results.model.eq(model)&results.degree_group.eq(group)&results.variable.eq(variable)]
            if len(selected) and selected.status.iloc[0]=='estimated':
                row=selected.iloc[0];coefficient[f'Q{group}']=f'{row.coefficient:.6f}{BASE["stars"](row.p_value)}'
                tstat[f'Q{group}']=f'({row.t_stat:.2f})'
            else:coefficient[f'Q{group}']='NA';tstat[f'Q{group}']=''
        rows.extend([coefficient,tstat])
    for label,value in [('Day FE','Yes'),('Firm FE','No'),('PERMNO clustered SE','Yes')]:
        rows.append({'variable':label,**{f'Q{i}':value for i in range(1,6)}})
    details={i['degree_group']:i for i in infos if i['model']==model}
    for label,key in [('Observations','n_obs'),('Firms','n_firms'),('Trading days','n_dates'),('Adj. R2','adj_r_squared')]:
        rows.append({'variable':label,**{f'Q{i}':details[i].get(key,'NA') for i in range(1,6)}})
    pd.DataFrame(rows).to_csv(out/f'05_02_{model.lower().replace(" ","_")}_table.csv',index=False,encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    panel=args.panel or BASE['latest_panel']()
    panel_manifest=json.loads((panel.parent/'run_summary.json').read_text(encoding='utf-8'))
    network=Path(panel_manifest['network_run'])
    if not network.is_absolute():network=ROOT/network
    manifest=json.loads((network/'run_summary.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or manifest.get('node_unit')!='permno':raise ValueError('Need completed PERMNO network')
    out=args.output_dir or ROOT/'Data/process/05_02_degree_group_regression'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    out.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();log=[];assignments=[];monthly=[]
    input_stats={str(panel):(panel.stat().st_size,panel.stat().st_mtime_ns)}
    for file in sorted(network.glob('????-??/nodes.parquet')):
        input_stats[str(file)]=(file.stat().st_size,file.stat().st_mtime_ns)
        groups=degree_groups(pd.read_parquet(file,columns=['permno','degree']),file.parent.name)
        assignments.append(groups)
        split_ties=groups.groupby('degree').degree_group.nunique().gt(1)
        for group in range(1,6):
            subset=groups.loc[groups.degree_group.eq(group),'degree']
            monthly.append({'network_month':file.parent.name,'degree_group':group,'n_stocks':len(subset),
                            'degree_mean':subset.mean(),'degree_median':subset.median(),
                            'degree_min':subset.min(),'degree_max':subset.max(),
                            'degree_values_split_across_groups':int(split_ties.sum())})
    assignments=pd.concat(assignments,ignore_index=True)
    if assignments.duplicated(['network_month','permno']).any():raise ValueError('Duplicate stock-month assignment')
    assignments.to_parquet(out/'05_02_monthly_degree_groups.parquet',index=False,compression='zstd')
    pd.DataFrame(monthly).to_csv(out/'05_02_monthly_group_diagnostics.csv',index=False,encoding='utf-8-sig')
    frame,source_counts=load_grouped_panel(panel,assignments,log)
    print('Grouped candidates:',BASE['sample_details'](frame),'\nOutput:',out,flush=True)
    all_infos=[];all_rows=[];periods={}
    for period in ['full_sample','2019_2020']:
        sample=frame if period=='full_sample' else frame.loc[frame.network_month.str[:4].isin(['2019','2020'])]
        period_out=out/period;period_out.mkdir()
        periods[period]=BASE['sample_details'](sample)
        infos=[];rows=[]
        for group in range(1,6):
            group_frame=sample.loc[sample.degree_group.eq(group)]
            for model in ['Model 1','Model 2','Model 3']:
                fit_rows,info,used=BASE['estimate'](group_frame,model,BASE['MODELS'][model],log)
                info.update(period=period,degree_group=group)
                info['sample_degree_min']=int(used.degree.min()) if len(used) else None
                info['sample_degree_max']=int(used.degree.max()) if len(used) else None
                for row in fit_rows:row.update(period=period,degree_group=group)
                infos.append(info);rows.extend(fit_rows)
                print(period,f'Q{group}',model,'N=',info['n_obs'],'beta=',info.get('reddit_coefficient'),'p=',info.get('reddit_p_value'),flush=True)
                del used
        results=pd.DataFrame(rows)
        results.to_csv(period_out/'05_02_degree_group_results.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{k:v for k,v in info.items() if k!='missing_by_variable'} for info in infos]).to_csv(period_out/'05_02_regression_diagnostics.csv',index=False,encoding='utf-8-sig')
        for model in ['Model 1','Model 2','Model 3']:paper_table(results,infos,model,period_out)
        (period_out/'05_02_summary.txt').write_text(
            'Monthly quintiles of positive-degree network stocks; Q1 lowest, Q5 highest.\n'
            'Ties split deterministically by PERMNO. Assignment fixed within month and independent of returns.\n'
            'Day FE, PERMNO-clustered CR1 SE including FE degrees of freedom, p-values t(G-1).\n'
            'Both dependent and return regressors in percentage points; other 05_01 settings unchanged.\n'
            'Complete cases separately for each model/group. Model 3 adds illiquidity only; no idio_vol/log_me, imputation, winsorization or portfolio sorts.\n'
            +json.dumps({'period_sample':periods[period],'models':infos},ensure_ascii=False,indent=2),encoding='utf-8')
        all_infos.extend(infos);all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(out/'05_02_all_results.csv',index=False,encoding='utf-8-sig')
    for file,stat in input_stats.items():
        if stat!=(Path(file).stat().st_size,Path(file).stat().st_mtime_ns):raise ValueError('Input changed during run')
    report={'status':'complete','panel':str(panel),'panel_sha256':BASE['hash_file'](panel),'network_run':str(network),
            'periods':periods,'source_counts':source_counts,'models':all_infos,'warnings':log,
            'grouping':'monthly positive-degree native network nodes; ascending degree then PERMNO; floor(rank0*5/n)+1',
            'group_population':'all positive-degree network nodes, before daily returns or complete-case selection',
            'grouping_uses_future_information':False,'model_implementation':'05_01_baseline_regression.estimate',
            'runtime_seconds':time.perf_counter()-tick,'output_directory':str(out)}
    (out/'run_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished:',out,flush=True)


if __name__=='__main__':main()
