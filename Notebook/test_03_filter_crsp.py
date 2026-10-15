import csv
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location('filter_crsp', Path(__file__).with_name('03_filter_crsp.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PermnoFilterTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / 'Data/raw/stock'
        raw.mkdir(parents=True)
        rows = [dict(zip(module.COLUMNS, [ticker, permno, '2012-01-03', '-10', '', '.A', '0', '100']))
                for ticker, permno in [('OLD', '1'), ('NEW', '1'), ('NEW', '2'), ('', '3'), ('OTHER', '4')]]
        pd.DataFrame(rows).to_csv(raw / 'RAW_crsp.csv', index=False)
        run = root / 'Data/process/02_02_filtered_comment_ticker/2026-10-15_00-00-00_UTC'
        run.mkdir(parents=True)
        pd.DataFrame({'permno': [1, 1, 3]}).to_parquet(run / 'comment_ticker_cleaned.parquet', index=False)
        review = root / 'review.csv'
        pd.DataFrame({'permno': [1, 2], 'ticker': ['NEW', 'NEW'], 'times': [2, 1], 'del': [0, 1]}).to_csv(review, index=False)
        info = {'status': 'complete', 'review_unit': 'permno', 'review_file': str(review),
                'review_sha256': module.sha256_file(review), 'excluded_permnos': [2]}
        manifest = run / 'run_summary.json'
        manifest.write_text(json.dumps(info), encoding='utf-8')
        return rows, manifest, review

    def test_aliases_shared_ticker_and_unmentioned_stock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _, _ = self.fixture(root)
            with redirect_stdout(io.StringIO()):
                result = module.run(root, chunk_size=2)
            with Path(result['output']).open(encoding='utf-8', newline='') as handle:
                self.assertEqual(list(csv.DictReader(handle)), [rows[0], rows[1], rows[3]])
            self.assertEqual(result['retained_permnos'], [1, 3])
            self.assertEqual((result['before_rows'], result['after_rows']), (5, 3))

    def test_changed_review_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, review = self.fixture(root)
            review.write_text(review.read_text().replace('NEW', 'CHANGED'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Review differs'):
                module.run(root)


if __name__ == '__main__':
    unittest.main()
