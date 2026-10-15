"""Test only 02_02 Table 1 functions, without modifying pipeline data."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd


class Table1Tests(unittest.TestCase):
    def test_date_filter_recomputes_units_and_preserves_aliases(self):
        path=Path(__file__).with_name('02_02_filter_reviewed_tickers.ipynb')
        notebook=json.loads(path.read_text(encoding='utf-8'))
        tree=ast.parse(''.join(notebook['cells'][5]['source']))
        ns={'pd':pd,'np':np,'ac_source':Path('fixture.parquet'),'ac_author_sources':[]}
        functions=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[])
        exec(compile(functions,'table1-functions','exec'),ns)
        rows=[('f',1,'t','A',1,'2018-06-30'),('f',1,'t','A',1,'2018-06-30'),
              ('f',2,'t','A',2,'2018-07-01'),('f',3,'u','[deleted]',3,'2018-07-02'),
              ('f',4,'','B',4,'2018-07-03'),('f',4,'','B',5,'2018-07-03')]
        data=pd.DataFrame(rows,columns=['source_file','source_row','thread_id','author','permno','comment_date'])
        data['comment_date']=pd.to_datetime(data.comment_date,utc=True)
        data['known_author']=~data.author.isin(['[deleted]'])
        ns['ac_comments']=data[['source_file','source_row','comment_date']].drop_duplicates()
        full=ns['table1_sample'](data,'full_sample')
        late=ns['table1_sample'](data,'2018_end_sample','2018-07-01')
        self.assertEqual(full[3].set_index('thread_id').loc['t','n_permnos'],2)
        self.assertEqual(late[3].set_index('thread_id').loc['t','n_permnos'],1)
        self.assertEqual(full[4].set_index('author').loc['A','n_permnos'],2)
        self.assertEqual(late[4].set_index('author').loc['A','n_permnos'],1)
        self.assertEqual(full[2]['unique_comments'],4)
        self.assertEqual(late[2]['unique_comments'],3)
        self.assertEqual(late[2]['min_date'],'2018-07-01')
        self.assertEqual(late[2]['max_date'],'2018-07-03')
        self.assertEqual(full[1].set_index('metric').loc['threads_mentioning_ge2_permnos_pct','percentage'],50)
        self.assertEqual(late[1].set_index('metric').loc['threads_mentioning_ge2_permnos_pct','percentage'],0)
        self.assertAlmostEqual(late[1].set_index('metric').loc['comments_mentioning_ge2_permnos_pct','percentage'],100/3)
        for result in [full,late]:
            panel_a,panel_b,metadata,*_=result
            self.assertTrue(np.allclose(panel_a['std'],panel_a['Std. Dev.'],equal_nan=True))
            self.assertTrue(np.allclose(panel_a.p90,panel_a.P90,equal_nan=True))
            self.assertTrue(np.allclose(panel_a['max'],panel_a.Max,equal_nan=True))
            self.assertTrue(panel_b.percentage.between(0,100).all())
            json.dumps(metadata,allow_nan=False)
        distribution=ns['activity_distribution']('x',[1,2,3,4,5])
        self.assertEqual(distribution['Min'],1)
        self.assertEqual(distribution['P10'],1.4)
        empty=ns['table1_sample'](data,'empty','2030-01-01')
        self.assertEqual(empty[2]['unique_comments'],0)
        self.assertTrue(empty[1].percentage.isna().all())
        json.dumps(empty[2],allow_nan=False)


if __name__=='__main__':unittest.main()
