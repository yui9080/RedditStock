"""Check fixed token matching and the 01 -> 02 -> 02_01 interface."""
import ast
from contextlib import redirect_stdout
from functools import lru_cache
import io
import json
from pathlib import Path
import re
import sqlite3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from test_01_month_exclusion import RawMonthExclusionTests


class TokenPipelineTests(RawMonthExclusionTests):
    def test_cleaning_to_matching_to_frequency(self):
        import csv
        dates = ['2012-04-15T12:00:00+00:00', '2012-05-01T00:00:00+00:00',
                 '2020-07-31T23:59:59+00:00', '2020-08-06T00:00:00+00:00']
        source = self.root / 'comments.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.env['COLUMNS'])
            writer.writeheader()
            writer.writerows(self.row(date) for date in dates)
        self.env['clean_csv'](source)
        cleaned = self.root / 'out/comments.csv'
        env = self.env
        env.update(pd=pd, pa=pa, pq=pq, re=re, lru_cache=lru_cache,
                   TICKER_SET={'ABC'}, CHUNK_SIZE=1, display=lambda *args: None)
        env['ticker_history'] = pd.DataFrame([('ABC', '2012-01-01', '2020-12-31', 12345)],
                                           columns=['ticker', 'secinfostartdt', 'secinfoenddt', 'permno'])
        run = self.root / 'matching'
        run.mkdir()
        env['RUN_DIR'] = run
        env['db'] = sqlite3.connect(':memory:')
        self.addCleanup(env['db'].close)
        env['db'].execute('CREATE TABLE uniques (file TEXT, stage TEXT, kind TEXT, value TEXT, PRIMARY KEY(file,stage,kind,value))')
        notebook = json.loads(Path(__file__).with_name('02_ticker_matching.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
        for forbidden in ['ahocorasick', 'BENCHMARK_', 'MATCHERS', 'benchmark_table']:
            self.assertNotIn(forbidden, code)
        names = {'TOKEN_RE', 'COMMENT_SCHEMA', 'MENTION_SCHEMA', 'AUDIT_SCHEMA'}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            text = ''.join(cell['source'])
            if 'def make_token_matcher' in text:
                with redirect_stdout(io.StringIO()):
                    exec(compile(text, '<matching engine>', 'exec'), env)
                continue
            tree = ast.parse(text)
            nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     or isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id in names for target in node.targets)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<matching>', 'exec'), env)
        self.assertEqual(env['SELECTED_METHOD'], 'token_hash_set')
        with redirect_stdout(io.StringIO()):
            stats = env['process_file'](cleaned)
        self.assertEqual(stats['matched_comment_count'], 2)
        mentions = pq.read_table(run / 'comments.mentions.parquet').to_pandas()
        self.assertEqual(mentions.year_month.tolist(), ['2012-05', '2020-07'])
        self.assertEqual(mentions.permno.tolist(), [12345, 12345])
        review = json.loads(Path(__file__).with_name('02_01_ticker_frequency_review.ipynb').read_text(encoding='utf-8'))
        for cell in review['cells']:
            if cell['cell_type'] == 'code':
                nodes = [node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'ranked_review']
                exec(compile(ast.Module(body=nodes, type_ignores=[]), '<review>', 'exec'), env)
        ranked = env['ranked_review']({12345: len(mentions)}, {12345: {'ABC'}})
        self.assertEqual(ranked.iloc[0].to_dict(), {'permno': 12345, 'ticker': 'ABC', 'times': 2, 'del': 1})


if __name__ == '__main__':
    import unittest
    unittest.main()
