import importlib.util
from pathlib import Path
import unittest
import tempfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

spec=importlib.util.spec_from_file_location('baseline',Path(__file__).with_name('05_01_baseline_regression.py'))
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)


class RegressionTests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(42)
        firms=np.repeat(np.arange(1,31),20)
        days=np.tile(pd.bdate_range('2020-01-01',periods=20),30)
        x=rng.normal(size=600)
        day_effect=np.tile(rng.normal(size=20),30)
        firm_noise=np.repeat(rng.normal(size=30),20)
        y=.04*x+day_effect+firm_noise+rng.normal(scale=.1,size=600)
        return pd.DataFrame({'permno':firms,'date':days,'reddit_ret_pct':x,'ret_t1_pct':y})

    def test_absorbed_day_fe_against_explicit_dummy_ols(self):
        f=self.sample()
        rows,info,_=baseline.estimate(f,'Model 1',['reddit_ret_pct'],[])
        actual=next(r for r in rows if r['variable']=='reddit_ret_pct')
        dummy=pd.get_dummies(f.date,drop_first=True,dtype=float)
        x=sm.add_constant(pd.concat([f[['reddit_ret_pct']],dummy],axis=1))
        result=sm.OLS(f.ret_t1_pct,x).fit(cov_type='cluster',cov_kwds={'groups':f.permno,'use_correction':True})
        self.assertAlmostEqual(actual['coefficient'],result.params['reddit_ret_pct'],places=10)
        self.assertAlmostEqual(actual['std_error'],result.bse['reddit_ret_pct'],places=9)
        self.assertAlmostEqual(info['adj_r_squared'],result.rsquared_adj,places=10)

    def test_complete_cases_and_unestimated_model3(self):
        f=self.sample();f.loc[0,'ret_t1_pct']=np.nan;f['idio_vol']=np.nan
        rows,info,_=baseline.estimate(f,'Model 1',['reddit_ret_pct'],[])
        self.assertEqual(info['n_obs'],599)
        rows,info,_=baseline.estimate(f,'Model 3',['reddit_ret_pct','idio_vol'],[])
        self.assertEqual(info['status'],'not_estimable');self.assertEqual(info['n_obs'],0)
        self.assertTrue(all(np.isnan(r['coefficient']) for r in rows))

    def test_percent_interpretation(self):
        s=baseline.interpretation({'status':'estimated','reddit_coefficient':.04})
        self.assertIn('0.04 percentage-point change',s)
        self.assertIn('equivalent to 4%',s)

    def test_model3_illiquidity_without_idio_or_market_cap(self):
        f=self.sample();rng=np.random.default_rng(31)
        for name in ['ret_d_pct','ret_20_2_pct','ret_120_21_pct']:f[name]=rng.normal(size=len(f))
        f['illiquidity']=rng.uniform(.01,1,len(f))
        variables=baseline.MODELS['Model 3']
        self.assertEqual(variables,baseline.MODELS['Model 2']+['illiquidity'])
        rows,info,_=baseline.estimate(f,'Model 3',variables,[])
        self.assertEqual(info['status'],'estimated');self.assertEqual(info['n_obs'],len(f))

    def test_network_year_filter_excludes_unavailable_months(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=pd.DataFrame({'permno':[1]*4,'date':pd.to_datetime(['2018-06-01','2019-01-02','2020-09-01','2020-10-01']),
                            'network_month':['2018-06','2019-01','2020-09','2020-10'],
                            'network_window_end':pd.to_datetime(['2018-05-31','2018-12-31','2020-08-31',None])})
            for variable in baseline.RETURNS+['illiquidity','idio_vol','log_me']:f[variable]=.01
            p=Path(tmp)/'panel.parquet';f.to_parquet(p,index=False)
            result,total=baseline.load_sample(p,pd.Timestamp('2018-01-01'),pd.Timestamp('2020-12-31'),[],network_years=(2019,2020))
            self.assertEqual(result.network_month.tolist(),['2019-01','2020-09'])
            self.assertEqual(total,4);self.assertEqual(result.attrs['excluded_network_filter'],2)


if __name__=='__main__':unittest.main()
