from predictml.demo import examples_dir
"""Causality, recursive evaluation, saved context and purchase semantics."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from predictml.scenarios.demand import (FEATURES, feature_vector, feature_table, forecast, evaluate,
                              validate_monthly, load_monthly, fit_demand)
from predictml.config import TrainingConfig
from predictml.workflow import run
from predictml.scenarios.serviceflow import prepare

ROOT = Path(__file__).resolve().parents[1]


def panel():
    return pd.DataFrame([dict(service_id=sid, year=p.year, month=p.month,
                              total_bookings=scale*(i+1))
                         for sid, scale in [(1, 1), (2, 10)]
                         for i, p in enumerate(pd.period_range('2023-01', '2024-12', freq='M'))])


class EchoPredictor:
    features = FEATURES
    def __init__(self):
        self.inputs = []
    def predict(self, values):
        self.inputs.append(values.copy())
        return values[:, :1]+1


class DemandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_months_cutoff_and_required_units(self):
        raw = self.root/'raw'
        shutil.copytree(examples_dir()/'raw', raw)
        prepare(raw, self.root/'prepared', '2026-11-14')
        frame, meta = load_monthly(self.root/'prepared')
        self.assertEqual(meta['last_complete_month'], '2026-10')
        self.assertEqual(frame[(frame.year == 2026)].month.max(), 10)
        self.assertFalse((self.root/'prepared/material_forecast.csv').exists())
        # v2 keeps fractional stock/norms instead of the v1 integer boundary.
        stock = pd.read_csv(raw/'materials.csv')
        stock['quantity'] = 0.2
        stock['purchase_step'] = 0.25
        stock.to_csv(raw/'materials.csv', index=False)
        norms = pd.read_csv(raw/'service_materials.csv')
        norms['required_quantity'] = 0.1
        norms.to_csv(raw/'service_materials.csv', index=False)
        prepare(raw, self.root/'fractional', '2026-11-14')
        self.assertEqual(json.loads((self.root/'fractional/materials.json').read_text())[0]['current_quantity'], 0.2)
        self.assertEqual(json.loads((self.root/'fractional/service_materials.json').read_text())[0]['required_quantity'], 0.1)
        # Neither unfinished November nor future December may change any output.
        notes = pd.read_csv(raw/'notes.csv')
        future = notes.start_at.str[:7] >= '2026-11'
        notes = pd.concat([notes, notes[future]], ignore_index=True)
        notes.to_csv(raw/'notes.csv', index=False)
        prepare(raw, self.root/'changed', '2026-11-14')
        pd.testing.assert_frame_equal(frame, load_monthly(self.root/'changed')[0])
        materials = pd.read_csv(raw/'materials.csv').drop(columns='unit')
        materials.to_csv(raw/'materials.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'materials.unit'):
            prepare(raw, self.root/'invalid', '2026-11-14')

    def test_future_and_other_series_cannot_change_training_features(self):
        frame = panel()
        before = feature_table(frame)
        changed = frame.copy()
        changed.loc[(changed.year == 2024) & (changed.month >= 7), 'total_bookings'] = 999999
        after = feature_table(changed)
        pd.testing.assert_frame_equal(before[before.ym < 2024*12+7], after[after.ym < 2024*12+7])
        changed = frame.copy()
        changed.loc[changed.service_id == 2, 'total_bookings'] *= 100
        pd.testing.assert_frame_equal(before[before.service_id == 1], feature_table(changed).query('service_id == 1'))
        row = before[(before.service_id == 1) & (before.year == 2023) & (before.month == 5)].iloc[0]
        self.assertEqual(row.demand_lag_1, 4)
        self.assertEqual(row.demand_mean_3, 3)
        self.assertEqual(row.demand_trend, 3)
        self.assertEqual(feature_vector([0, 0, 0, 0, 0, 0], 1)[4], 1)

    def test_recursion_uses_predictions_not_future_actuals(self):
        frame = panel()
        predictor = EchoPredictor()
        result = forecast(frame, predictor, 3)
        first = [r for r in result if r['service_id'] == 1]
        self.assertEqual([r['predicted_bookings'] for r in first], [25, 26, 27])
        self.assertEqual([(r['year'], r['month']) for r in first], [(2025, 1), (2025, 2), (2025, 3)])
        self.assertEqual(predictor.inputs[1][0, 0], 25)
        for horizon in (1, 2, 3):
            short = forecast(frame, EchoPredictor(), horizon)
            self.assertEqual(short, [r for r in result if r['lead'] <= horizon])
        start = 2024*12+10
        a = evaluate(frame, EchoPredictor(), start)
        changed = frame.copy()
        changed.loc[(changed.year == 2024) & (changed.month >= 10), 'total_bookings'] = 500
        b = evaluate(changed, EchoPredictor(), start)
        self.assertEqual([r['predicted_bookings'] for r in a['models']['mlp']['rows']],
                         [r['predicted_bookings'] for r in b['models']['mlp']['rows']])
        self.assertNotEqual(a['models']['mlp']['metrics'], b['models']['mlp']['metrics'])
        self.assertEqual(set(a['models']), {'mlp', 'last_value', 'mean3'})

    def test_zero_demand_and_zero_denominator(self):
        frame = panel()
        frame['total_bookings'] = 0
        class Negative(EchoPredictor):
            def predict(self, values):
                return np.array([[-5.]])
        self.assertTrue(all(r['predicted_bookings'] == 0 for r in forecast(frame, Negative(), 3)))
        report = evaluate(frame, Negative(), 2024*12+10)
        metrics = report['models']['mlp']['metrics']
        self.assertEqual(metrics['mae'], 0)
        self.assertIsNone(metrics['wmape'])
        self.assertIsNone(metrics['r2'])

    def test_invalid_or_incomplete_calendar(self):
        frame = panel()
        with self.assertRaisesRegex(ValueError, 'complete common monthly calendar'):
            validate_monthly(frame.drop(index=4))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_monthly(pd.concat([frame, frame.iloc[:1]]))

    def test_train_stats_and_weights_ignore_test_targets(self):
        frame = panel()
        changed = frame.copy()
        changed.loc[(changed.year == 2024) & (changed.month >= 10), 'total_bookings'] = 999
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        rng = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng)
        metadata = {'as_of': '2025-01-01'}
        torch.manual_seed(17)
        a = fit_demand(frame, metadata, TrainingConfig(epochs=100))
        torch.manual_seed(17)
        b = fit_demand(changed, metadata, TrainingConfig(epochs=100))
        self.assertEqual(a['feature_mean'], b['feature_mean'])
        self.assertEqual(a['feature_std'], b['feature_std'])
        self.assertEqual(a['metrics']['best_val_loss'], b['metrics']['best_val_loss'])
        for key, tensor in a['model_state'].items():
            torch.testing.assert_close(tensor, b['model_state'][key], rtol=0, atol=0)
        self.assertEqual(a['demand_context']['split']['test']['start'], '2024-10')
        self.assertEqual(a['metrics']['units'], 'completed_bookings')
        expected_train = feature_table(frame)
        expected_train = expected_train[expected_train.ym < 2024*12+7]
        np.testing.assert_array_equal(np.array(a['feature_mean'], dtype=np.float32),
                                      expected_train[list(FEATURES)].values.astype(np.float32).mean(axis=0))
        self.assertEqual(a['metrics']['best_val_loss'],
                         a['metrics']['recursive_validation']['models']['mlp']['metrics']['mse'])

    def test_saved_model_forecasts_in_new_process_without_history_file(self):
        raw = examples_dir()/'raw'
        data, output = self.root/'prepared', self.root/'trained'
        prepare(raw, data, '2027-01-01')
        result = run('materials', data, output, epochs=50, months=2)
        model = output/'material_model.pth'
        catalog = self.root/'catalog'
        catalog.mkdir()
        for name in ('materials.json', 'service_materials.json'):
            shutil.copyfile(data/name, catalog/name)
        destination = self.root/'reloaded'
        subprocess.run([sys.executable, '-m', 'predictml', 'predict', 'materials',
                        '--data', str(catalog), '--output', str(destination), '--model', str(model), '--months', '2'],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertEqual(result, json.loads((destination/'material_predictions.json').read_text()))
        self.assertEqual(json.loads((output/'demand_predictions.json').read_text()),
                         json.loads((destination/'demand_predictions.json').read_text()))
        # Extend history with a newly completed month without refitting scaling.
        frame = pd.read_csv(data/'monthly_demand.csv')
        appended = pd.DataFrame([dict(service_id=s, year=2027, month=1, total_bookings=12) for s in (1, 2)])
        extended = pd.concat([frame, appended], ignore_index=True)
        extended.to_csv(data/'monthly_demand.csv', index=False)
        meta = json.loads((data/'preparation.json').read_text())
        meta.update(as_of='2027-02-01', last_complete_month='2027-01')
        (data/'preparation.json').write_text(json.dumps(meta))
        run('materials', data, self.root/'extended', train=False, model=model, months=1)
        rows = json.loads((self.root/'extended/demand_predictions.json').read_text())
        self.assertEqual({(r['year'], r['month']) for r in rows}, {(2027, 2)})
        extended.loc[0, 'total_bookings'] += 1
        extended.to_csv(data/'monthly_demand.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'saved history must be unchanged'):
            run('materials', data, self.root/'changed-history', train=False, model=model)
        state = torch.load(model, weights_only=False)
        self.assertEqual(state['demand_context']['contract_version'], 2)
        self.assertEqual(len(state['demand_context']['history']), 96)
        state.pop('demand_context')
        torch.save(state, self.root/'old.pth')
        with self.assertRaisesRegex(ValueError, 'context required'):
            run('materials', catalog, self.root/'invalid', train=False, model=self.root/'old.pth')

