from predictml.demo import examples_dir
"""Contract checks run with unittest or pytest; no training required."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from predictml.scenarios.serviceflow import prepare_legacy as prepare


class LegacyPurchaseTests(unittest.TestCase):
    def test_fixed_demand_purchase_and_year_rollover(self):
        import torch
        from unittest.mock import patch
        from predictml.workflow import run as current_run

        def run(*args, **kwargs):
            return current_run(*args, mode="legacy-parity", **kwargs)
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
            prepare(examples_dir()/'raw', root/'data', '2026-12-31')
            path = root/'data/material_forecast.csv'
            frame = pd.read_csv(path)
            frame['avg_bookings'] = 10.0
            frame.to_csv(path, index=False)
            checkpoint = root/'model.pth'
            checkpoint.touch()
            state = dict(input_size=7, hidden_sizes=[32, 16], model_state={},
                         feature_mean=[0]*7, feature_std=[1]*7, features=list(MATERIALS.features))
            with patch('predictml.models.MLP', ConstantModel), patch('predictml.models.torch.load', return_value=state):
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
        from predictml.scenarios.serviceflow import ruby_round
        self.assertEqual(ruby_round(2.675, 2), 2.68)
        self.assertEqual(ruby_round(-2.675, 2), -2.68)

