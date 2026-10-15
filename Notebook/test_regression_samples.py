"""Keep all regression stages on the same predictor-date sample definitions."""
from pathlib import Path
import json
import runpy
import tempfile
import unittest
import pandas as pd

NOTEBOOK=Path(__file__).resolve().parent
EXPECTED={'full_sample':('2012-11-01','2020-08-31'),
          'late_sample':('2019-01-01','2020-08-31')}


class SampleDateTests(unittest.TestCase):
    def test_shared_periods_in_all_stages(self):
        for filename in ['05_build_regression_panel.py','05_01_baseline_regression.py',
                         '05_02_degree_group_regression.py','05_03_degree_heterogeneity_test.py']:
            module=runpy.run_path(str(NOTEBOOK/filename))
            self.assertEqual(module['PERIODS'],EXPECTED,filename)

    def test_sample_boundaries_in_loader(self):
        module=runpy.run_path(str(NOTEBOOK/'05_01_baseline_regression.py'))
        dates=pd.to_datetime(['2012-10-31','2012-11-01','2018-12-31',
                              '2019-01-01','2020-08-31','2020-09-01'])
        frame=pd.DataFrame({'permno':1,'date':dates})
        for variable in module['RETURNS']+['illiquidity']:frame[variable]=.01
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'panel.parquet'
            frame.to_parquet(path,index=False)
            for name,(start,end) in EXPECTED.items():
                actual,_=module['load_sample'](path,pd.Timestamp(start),pd.Timestamp(end),[])
                expected=frame.loc[frame.date.between(start,end),'date'].tolist()
                self.assertEqual(actual.date.tolist(),expected,name)

    def check_bounds(self,name,fit):
        self.assertGreaterEqual(fit['min_date'],EXPECTED[name][0])
        self.assertLessEqual(fit['max_date'],EXPECTED[name][1])


if __name__=='__main__':unittest.main()
