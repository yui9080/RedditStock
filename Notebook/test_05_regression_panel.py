"""Scientific timing and algebra regression tests, using only synthetic inputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('panel05', Path(__file__).with_name('05_build_regression_panel.py'))
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


class PanelTests(unittest.TestCase):
    def test_compounding_exact_windows_missing_and_total_loss(self):
        r = pd.Series(np.full(130,.01))
        a = panel.compounded_window(r,2,19)
        b = panel.compounded_window(r,21,100)
        self.assertTrue(a.iloc[:20].isna().all())
        self.assertAlmostEqual(a.iloc[20],1.01**19-1)
        self.assertTrue(b.iloc[:120].isna().all())
        self.assertAlmostEqual(b.iloc[120],1.01**100-1)
        r.iloc[2] = -1
        self.assertEqual(panel.compounded_window(r,2,19).iloc[20],-1)
        r.iloc[3] = np.nan
        self.assertTrue(np.isnan(panel.compounded_window(r,2,19).iloc[20]))
        cleaned = panel.return_values(pd.Series(['A','-66','-1','0','.2','inf']))
        self.assertTrue(cleaned.iloc[:2].isna().all())
        self.assertEqual(cleaned.iloc[2],-1)

    def test_source_duplicate_cleanup_and_conflict_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'crsp.csv'
            row={'permno':1,'DlyCalDt':'2020-01-02','DlyRet':.1,'DlyPrc':10,
                 'DlyCap':100,'DlyVol':20,'HdrCUSIP':'ABC'}
            pd.DataFrame([row,row]).to_csv(path,index=False)
            data,info=panel.load_crsp(path,'thousand_usd',[])
            self.assertEqual(len(data),1)
            self.assertEqual(info['exact_duplicate_rows_dropped'],1)
            other=dict(row,HdrCUSIP='DIFFERENT')
            pd.DataFrame([row,other]).to_csv(path,index=False)
            with self.assertRaises(ValueError):panel.load_crsp(path,'thousand_usd',[])

    def network(self, root):
        folder = root/'2020-10';folder.mkdir()
        pd.DataFrame({'permno_i':[1,1], 'permno_j':[2,3],
                      'n_common_threads':[1,3], 'weight':np.log1p([1,3])}).to_parquet(folder/'edges.parquet',index=False)
        pd.DataFrame({'permno':[1,2,3,4]}).to_parquet(folder/'nodes.parquet',index=False)
        timing = pd.DataFrame([{'network_month':'2020-10','window_start':'2020-04','window_end':'2020-09'}]).set_index('network_month')
        return folder,timing

    def test_peer_weights_renormalization_and_no_self_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,timing=self.network(Path(tmp))
            graph=panel.validate_network(folder,timing)
            month=pd.DataFrame({'permno':[1,2,3,4,1,2,3,4],
                                'date':pd.to_datetime(['2020-10-01']*4+['2020-10-02']*4),
                                'ret_d':[.9,.1,.3,.5,.9,np.nan,.4,.5]})
            panel.reddit_returns(month,graph)
            self.assertAlmostEqual(month.reddit_ret.iloc[0],(.1+2*.3)/3)
            self.assertAlmostEqual(month.reddit_ret.iloc[4],.4)
            self.assertEqual(month.n_valid_peers_d.iloc[4],1)
            self.assertTrue(np.isnan(month.reddit_ret.iloc[3]))
            self.assertEqual(month.n_reddit_peers.iloc[3],0)
            timing.loc['2020-10','window_end']='2020-10'
            with self.assertRaises(ValueError):panel.validate_network(folder,timing)

    def test_industry_leave_one_out(self):
        f=pd.DataFrame({'date':pd.to_datetime(['2020-10-01']*3),'siccd':[1234]*3,
                        'ret_d':[.9,.1,.3],'me_weight_lag1':[100,200,300]})
        result=panel.industry_returns(f)
        self.assertAlmostEqual(result.iloc[0],.22)
        self.assertAlmostEqual(result.iloc[1],.45)
        f.loc[1,'ret_d']=np.nan
        self.assertAlmostEqual(panel.industry_returns(f).iloc[0],.3)

    def test_trading_sequence_and_monthend(self):
        dates=pd.bdate_range('2020-01-01',periods=130)
        f=pd.DataFrame({'permno':1,'date':dates,'ret_d':.01,'me_daily':np.arange(130)+100.,'prc':10.,'vol':100.})
        result=panel.add_stock_controls(f)
        friday=result.loc[result.date.eq(pd.Timestamp('2020-01-03'))].iloc[0]
        self.assertEqual(friday.ret_t1_date,pd.Timestamp('2020-01-06'))
        ends=panel.monthly_me(result)
        feb=ends.loc[ends.network_month.eq('2020-02')].iloc[0]
        self.assertEqual(feb.me_source_date,pd.Timestamp('2020-01-31'))
        self.assertTrue(result.ret_t1.tail(1).isna().all())
        self.assertTrue(result.illiquidity.iloc[:10].isna().all())
        self.assertAlmostEqual(result.illiquidity_raw.iloc[10],.01/1000)

    def test_factor_residualization_never_uses_future(self):
        rng=np.random.default_rng(42);dates=pd.bdate_range('2020-01-01',periods=35)
        x=rng.normal(0,.01,(35,3));noise=rng.normal(0,.001,35)
        factors=pd.DataFrame(x,index=dates,columns=['MKT_RF','SMB','HML']);factors['RF']=.0001
        f=pd.DataFrame({'permno':1,'date':dates,'ret_d':.0001+.001+x@np.array([1.,.5,.2])+noise})
        f['lag1_date']=f.date.shift(1)
        panel.add_idio(f,factors,'FF3',20,15)
        expected_x=np.column_stack([np.ones(20),x[:20]])
        y=f.ret_d.iloc[:20].to_numpy()-.0001
        beta=np.linalg.lstsq(expected_x,y,rcond=None)[0]
        self.assertAlmostEqual(f.idio_vol.iloc[20],np.std(y-expected_x@beta,ddof=1))
        baseline=f.idio_vol.iloc[20]
        f.loc[20:,'ret_d']=100.
        panel.add_idio(f,factors,'FF3',20,15)
        self.assertAlmostEqual(f.idio_vol.iloc[20],baseline)

    def test_book_equity_availability_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fundamentals.csv'
            pd.DataFrame({'permno':[1],'available_date':['2020-02-15'],'datadate':['2019-12-31'],
                          'seqq':[10.],'pstkq':[1.],'txditcq':[1.]}).to_csv(path,index=False)
            data=panel.prepare_fundamentals(path,'thousand_usd','million_usd',[])
            for cutoff,expected in [('2020-01-31',None),('2020-02-29',.5)]:
                f=pd.DataFrame({'permno':[1],'monthend_cutoff':pd.to_datetime([cutoff]),'me':[20000.]})
                panel.add_bm(f,data)
                if expected is None:self.assertTrue(np.isnan(f.bm.iloc[0]))
                else:self.assertAlmostEqual(f.bm.iloc[0],expected)

    def test_cli_writes_complete_panel_and_missing_control_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);net=root/'network';net.mkdir();folder,timing=self.network(net)
            timing.reset_index().to_csv(net/'monthly_network_summary.csv',index=False)
            (net/'run_summary.json').write_text(json.dumps({'status':'complete','node_unit':'permno'}))
            dates=pd.bdate_range('2020-09-01','2020-10-09')
            rows=[{'permno':p,'DlyCalDt':str(d.date()),'DlyRet':.01*p,'DlyPrc':10.,'DlyVol':100.,
                   'DlyCap':1000.*p,'siccd':1234} for p in [1,2,3,4] for d in dates]
            crsp=root/'crsp.csv';pd.DataFrame(rows).to_csv(crsp,index=False)
            args=['panel05','--network-run',str(net),'--crsp',str(crsp),'--output-dir',str(root/'output')]
            with patch('sys.argv',args):panel.main()
            f=pd.read_parquet(root/'output/04_regression_panel.parquet')
            self.assertEqual(len(f),len(rows));self.assertFalse(f.duplicated(['permno','date']).any())
            self.assertTrue(f.bm.isna().all() and f.idio_vol.isna().all())
            self.assertTrue(f.loc[f.network_month.eq('2020-09'),'reddit_ret'].isna().all())
            self.assertTrue(f.loc[f.network_month.eq('2020-10') & f.permno.eq(1),'reddit_ret'].notna().all())
            summary=json.loads((root/'output/run_summary.json').read_text())
            self.assertFalse(summary['regression_executed'])
            self.assertEqual(summary['diagnostics']['observations_with_all_baseline_variables'],0)
            # The timing checker must reject an injected future RHS cutoff.
            subset=f.loc[f.network_month.eq('2020-10')].copy()
            eligible=subset.illiquidity.notna()
            subset.loc[eligible,'illiquidity_max_date']=subset.loc[eligible,'date']
            with self.assertRaises(ValueError):panel.lookahead_checks(subset)


if __name__=='__main__':unittest.main()
