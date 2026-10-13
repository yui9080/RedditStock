import runpy
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm

TESTED=runpy.run_path(str(Path(__file__).with_name('05_03_degree_heterogeneity_test.py')))


class HeterogeneityTests(unittest.TestCase):
    def test_covariance_cross_terms_and_difference_cancellation(self):
        names=['const','reddit_ret_pct']+[f'reddit_ret_x_Q{i}' for i in range(2,6)]
        beta=np.array([0,.1,.02,.03,.04,.05]);cov=np.eye(6)*.01
        cov[1,5]=cov[5,1]=.004
        slopes,tests=TESTED['contrasts'](beta,cov,names,99)
        self.assertAlmostEqual(slopes[4]['slope'],.15)
        self.assertAlmostEqual(slopes[4]['std_error'],np.sqrt(.028))
        self.assertAlmostEqual(tests[1]['difference_Q5_Q1'],.05)
        self.assertAlmostEqual(tests[1]['std_error_difference'],.1)
        self.assertEqual(len(tests),2)

    def test_pooled_full_covariance_matches_dummy_ols(self):
        rng=np.random.default_rng(71);n=600
        f=pd.DataFrame({'permno':np.repeat(np.arange(30),20),
                        'date':np.tile(pd.bdate_range('2020-01-01',periods=20),30),
                        'degree_group':np.repeat(np.arange(30)%5+1,20),
                        'reddit_ret_pct':rng.normal(size=n)})
        f['ret_t1_pct']=f.reddit_ret_pct*(.1+.02*f.degree_group)+rng.normal(size=n)
        result,beta,cov,names=TESTED['interaction_fit'](f,['reddit_ret_pct'])
        x=f[['reddit_ret_pct']].copy();x.insert(0,'const',1.)
        for q in range(2,6):
            x[f'Q{q}']=f.degree_group.eq(q).astype(float)
            x[f'reddit_ret_x_Q{q}']=f.reddit_ret_pct*x[f'Q{q}']
        x=pd.concat([x,pd.get_dummies(f.date,drop_first=True,dtype=float)],axis=1)
        expected=sm.OLS(f.ret_t1_pct,x).fit(cov_type='cluster',cov_kwds={'groups':f.permno,'use_correction':True})
        selected=names[1:]  # Intercept normalization differs across FE parameterizations.
        self.assertTrue(np.allclose(result.params[selected],expected.params[selected],atol=1e-10))
        self.assertTrue(np.allclose(cov[1:,1:],expected.cov_params().loc[selected,selected],atol=1e-10))


if __name__=='__main__':unittest.main()
