"""Contract checks run with unittest or pytest; no training required."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import pandas as pd
from predictml.contracts import load_raw, validate_prepared
from predictml.preparation import prepare

ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name)/'raw'
        shutil.copytree(ROOT/'examples/raw', self.raw)
        self.output = Path(self.temp.name)/'prepared'

    def change(self, table, column, value):
        path = self.raw/f'{table}.csv'
        data = pd.read_csv(path)
        data[column] = data[column].astype(object)
        data.loc[0, column] = value
        data.to_csv(path, index=False)

    def test_reject_invalid_raw_values(self):
        cases = [('notes', 'service_id', 999), ('notes', 'client_id', None),
                 ('notes', 'start_at', 'invalid'), ('notes', 'status', 'other'),
                 ('services', 'active', 'maybe'), ('services', 'price', float('inf')),
                 ('materials', 'quantity', -1), ('materials', 'minimal_quantity', 1.5),
                 ('service_materials', 'material_id', 999),
                 ('service_materials', 'required_quantity', 'bad')]
        for table, column, value in cases:
            with self.subTest(table=table, column=column):
                shutil.copyfile(ROOT/'examples/raw'/f'{table}.csv', self.raw/f'{table}.csv')
                self.change(table, column, value)
                with self.assertRaisesRegex(ValueError, f'{table}.{column}'):
                    load_raw(self.raw)
                shutil.copyfile(ROOT/'examples/raw'/f'{table}.csv', self.raw/f'{table}.csv')

    def test_duplicate_ids_and_headers(self):
        path = self.raw/'services.csv'
        data = pd.read_csv(path)
        pd.concat([data, data.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_raw(self.raw)
        path.write_text('id,price,active,id\n1,10,true,1\n')
        with self.assertRaisesRegex(ValueError, 'duplicate column'):
            load_raw(self.raw)

    def test_prepared_validation(self):
        prepare(self.raw, self.output, '2026-12-31')
        validate_prepared(self.output, 'materials', True)
        validate_prepared(self.output, 'price', True)
        path = self.output/'material_forecast.csv'
        data = pd.read_csv(path)
        pd.concat([data, data.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_prepared(self.output, 'materials', True)

    def test_prepared_numeric_and_json_errors(self):
        prepare(self.raw, self.output, '2026-12-31')
        path = self.output/'price_analysis.csv'
        data = pd.read_csv(path)
        data['base_price'] = data['base_price'].astype(float)
        data.loc[0, 'base_price'] = float('inf')
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'base_price'):
            validate_prepared(self.output, 'price', True)
        path = self.output/'materials.json'
        original = json.loads(path.read_text())
        for value in ('20', None, -1):
            with self.subTest(price=value):
                original[0]['price'] = value
                path.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, 'materials.price'):
                    validate_prepared(self.output, 'materials', False)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'array of objects'):
            validate_prepared(self.output, 'materials', False)

    def test_insufficient_history_and_invalid_cutoff(self):
        with self.assertRaisesRegex(ValueError, 'four calendar months'):
            prepare(self.raw, self.output, '2023-02-01')
        with self.assertRaisesRegex(ValueError, 'as_of'):
            prepare(self.raw, self.output, 'bad')
        prepare(self.raw, self.output, '2023-07-01')
        with self.assertRaisesRegex(ValueError, 'extend history'):
            validate_prepared(self.output, 'materials', True)
        with self.assertRaisesRegex(ValueError, '7 training months'):
            validate_prepared(self.output, 'price', True)

    @unittest.skipUnless(shutil.which('ruby'), 'Ruby unavailable: reference comparison skipped')
    def test_actual_ruby_service_calculations(self):
        # Counts include zero months and incomplete previous three-month windows.
        # Repeated completed visits in the first month all count as first visits.
        counts = [2, 0, 4, 3, 1, 6, 2, 0]
        rows = []
        for month, count in enumerate(counts, 1):
            rows += [[1, 1, 2, f'2024-{month:02d}-{i+1:02d} 10:00:00', 100] for i in range(count)]
        # Canceled record must not affect aggregation or first completed visit.
        rows.append([1, 2, 1, '2024-01-20 10:00:00', 100])
        pd.DataFrame(rows, columns='service_id client_id status start_at total_price'.split()).to_csv(self.raw/'notes.csv', index=False)
        payload = {'services': [{'id': 1, 'price': 1000, 'active': True},
                                {'id': 2, 'price': 1500, 'active': True}],
                   'months': [[2024, m] for m in range(1, 9)], 'costs': {'1': 40.0, '2': 60.0},
                   'counts': [[1, 2024, m, n, n*100, 100 if n else 0, n if m == 1 else 0]
                              for m, n in enumerate(counts, 1)]}
        result = subprocess.run(['ruby', str(ROOT/'tests/reference/run.rb')],
                                input=json.dumps(payload), text=True, capture_output=True, check=True)
        expected = json.loads(result.stdout)
        prepare(self.raw, self.output, '2024-08-31')
        for name, filename in [('materials', 'material_forecast.csv'), ('price', 'price_analysis.csv')]:
            actual = pd.read_csv(self.output/filename)
            pd.testing.assert_frame_equal(actual, pd.DataFrame(expected[name], columns=actual.columns),
                                          check_dtype=False, atol=1e-9, rtol=0)


class LegacyPurchaseTests(unittest.TestCase):
    def test_fixed_demand_purchase_and_year_rollover(self):
        import torch
        from unittest.mock import patch
        from predictml.pipeline import run
        from predictml.config import MATERIALS

        class ConstantModel:
            def __init__(self, *args, **kwargs):
                pass
            def load_state_dict(self, state):
                pass
            def eval(self):
                pass
            def __call__(self, values):
                return torch.tensor([[1.0]])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(ROOT/'examples/raw', root/'data', '2026-12-31')
            path = root/'data/material_forecast.csv'
            frame = pd.read_csv(path)
            frame['avg_bookings'] = 10.0
            frame.to_csv(path, index=False)
            checkpoint = root/'model.pth'
            checkpoint.touch()
            state = dict(input_size=7, hidden_sizes=[32, 16], model_state={},
                         feature_mean=[0]*7, feature_std=[1]*7, features=list(MATERIALS.features))
            with patch('predictml.inference.MLP', ConstantModel), patch('predictml.inference.torch.load', return_value=state):
                for horizon in (1, 2, 3):
                    result = run('materials', root/'data', root/f'output-{horizon}',
                                 train=False, model=checkpoint, months=horizon)[0]
                    self.assertEqual(result['total_predicted_usage'], 50*horizon)
                    self.assertEqual(result['purchase_qty'], 50*horizon-5)
                    self.assertEqual(result['purchase_cost'], (50*horizon-5)*20)
                    self.assertEqual(result['months'][0]['year'], 2027)
                    self.assertEqual([m['month'] for m in result['months']], list(range(1, horizon+1)))
                # Stock covers demand and the safety stock: no negative purchase.
                materials = json.loads((root/'data/materials.json').read_text())
                materials[0]['current_quantity'] = 1000
                (root/'data/materials.json').write_text(json.dumps(materials))
                result = run('materials', root/'data', root/'covered', train=False,
                             model=checkpoint, months=3)[0]
                self.assertEqual(result['purchase_qty'], 0)
                self.assertEqual(result['purchase_cost'], 0)

    def test_ruby_rounding_boundary(self):
        from predictml.preparation import ruby_round
        self.assertEqual(ruby_round(2.675, 2), 2.68)
        self.assertEqual(ruby_round(-2.675, 2), -2.68)
