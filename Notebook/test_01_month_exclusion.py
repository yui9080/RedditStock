"""Test raw cleaning boundaries, audit reasons, and preservation of other rules."""
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import time
import unittest


class RawMonthExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(Path=Path, Counter=Counter, csv=csv, json=json, math=math,
                        datetime=datetime, timezone=timezone, time=time,
                        OUTPUT_DIR=self.root / 'out', PROGRESS_EVERY=0)
        notebook = json.loads(Path(__file__).with_name('01_Initial_CSV_Clean.ipynb').read_text(encoding='utf-8'))
        names = {'COLUMNS', 'REQUIRED_COLUMNS', 'EXCLUDED_UTC_MONTHS'}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                        or isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id in names for target in node.targets)]
            exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook>', 'exec'), self.env)

    def row(self, date):
        row = dict.fromkeys(self.env['COLUMNS'], '')
        row.update(author='valid_user', body='ABC', id='comment', link_id='t3_thread',
                   created_utc=str(datetime.fromisoformat(date).timestamp()))
        return row

    def test_utc_month_boundaries_and_existing_validation(self):
        checks = [('2012-03-31T23:59:59+00:00', None),
                  ('2012-04-01T00:00:00+00:00', '2012-04'),
                  ('2012-04-30T23:59:59.5+00:00', '2012-04'),
                  ('2012-05-01T00:00:00+00:00', None),
                  ('2020-07-31T23:59:59.5+00:00', None),
                  ('2020-08-01T00:00:00+00:00', '2020-08'),
                  ('2020-08-31T23:59:59+00:00', '2020-08'),
                  ('2020-09-01T00:00:00+00:00', None)]
        for date, month in checks:
            self.assertEqual(self.env['drop_reasons'](self.row(date)),
                             ['excluded_utc_month_' + month] if month else [])
        row = self.row('2020-07-01T00:00:00+00:00')
        self.assertEqual(self.env['drop_reasons'](dict(row, created_utc='NaN')), ['invalid_created_utc'])
        self.assertEqual(self.env['drop_reasons'](dict(row, link_id=' t3_ ')), ['empty_normalized_thread_id'])

    def test_cleaned_csv_and_drop_audit(self):
        dates = ['2012-04-15T12:00:00+00:00', '2012-05-01T00:00:00+00:00',
                 '2020-07-31T23:59:59+00:00', '2020-08-06T00:00:00+00:00']
        source = self.root / 'input.csv'
        rows = [self.row(date) for date in dates]
        with source.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.env['COLUMNS'])
            writer.writeheader()
            writer.writerows(rows)
        original = source.read_bytes()
        stats = self.env['clean_csv'](source)
        self.assertEqual((stats['input_records'], stats['rows'], stats['dropped_records']), (4, 2, 2))
        self.assertEqual(stats['drop_reason_counts'], {'excluded_utc_month_2012-04': 1, 'excluded_utc_month_2020-08': 1})
        with (self.root / 'out/input.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)), [rows[1], rows[2]])
        audit = [json.loads(line) for line in (self.root / 'out/input.dropped.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([record['source_row'] for record in audit], [1, 4])
        self.assertEqual(source.read_bytes(), original)

    def test_author_policy_and_required_fields_in_csv(self):
        valid = self.row('2020-07-01T00:00:00+00:00')
        bad = [dict(valid, **{name: value}) for name in ['author', 'body', 'created_utc']
               for value in ['', '  ']]
        bad += [dict(valid, author=value) for value in
                ['[deleted]', '[REMOVED]', ' Deleted ', 'removed', 'delete', '[remove]', '[delete]', 'REMOVE']]
        kept = [valid, dict(valid, author='deleted_user'), dict(valid, author='remove123')]
        source = self.root / 'authors.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.env['COLUMNS'])
            writer.writeheader()
            writer.writerows(bad + kept)
        stats = self.env['clean_csv'](source)
        self.assertEqual(stats['rows'], len(kept))
        self.assertEqual(stats['dropped_records'], len(bad))
        self.assertEqual(stats['drop_reason_counts'], {
            'missing_author': 2, 'missing_body': 2, 'missing_created_utc': 2,
            'deleted_author': 4, 'removed_author': 4})
        with (self.root / 'out/authors.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)), kept)


if __name__ == '__main__':
    unittest.main()
