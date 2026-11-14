import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from predictml.config import TrainingConfig
from predictml.training import fit_arrays
from predictml.metrics import regression_metrics
from predictml.tabular.application import train_table, train_file, TabularModel
from predictml.comparison import compare_service
from predictml.pipeline import run
from predictml.preparation import prepare

ROOT = Path(__file__).resolve().parents[1]
FAST = TrainingConfig(epochs=4, validation_interval=1, patience=2, seed=19)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_metric_known_values_and_undefined_cases(self):
        result = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(result['mae'], 2/3)
        self.assertAlmostEqual(result['mse'], 2/3)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2/3))
        self.assertAlmostEqual(result['wmape'], 100/3)
        self.assertEqual(result['r2'], 0)
        self.assertIsNone(regression_metrics([0, 0], [1, 1])['wmape'])
        self.assertIsNone(regression_metrics([1], [1])['r2'])
        for a, b in [([], []), ([1, 2], [1]), ([float('inf')], [1])]:
            with self.assertRaises(ValueError):
                regression_metrics(a, b)

    def test_early_stopping_restores_best_weights(self):
        x = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y = 2*x
        losses, snapshots = iter([5., 1., 3., 4., 9.]), []
        def objective(state):
            snapshots.append({k: v.clone() for k, v in state['model_state'].items()})
            return next(losses)
        log = {}
        weights, loss, _ = fit_arrays(x, y, x, y, x, (4,),
                                      TrainingConfig(epochs=10, validation_interval=1, patience=2),
                                      validation_evaluator=objective, mean=np.zeros(1), std=np.ones(1),
                                      features=['x'], training_log=log)
        self.assertEqual(log['epochs_ran'], 4)
        self.assertEqual(log['best_epoch'], 2)
        self.assertTrue(log['stopped_early'])
        self.assertEqual(loss, 1)
        for key in weights:
            torch.testing.assert_close(weights[key], snapshots[1][key], rtol=0, atol=0)

    def test_tabular_same_splits_baseline_and_linear_roundtrip(self):
        frame = pd.DataFrame({'x': np.arange(40), 'y': 3*np.arange(40)+5})
        source = self.root/'input.csv'
        frame.to_csv(source, index=False)
        model = train_file(source, self.root/'trained', {'target': 'y', 'seed': 19}, settings=FAST, model_kind='best')
        self.assertEqual(set(model.report['models']), {'mlp', 'linear', 'baseline'})
        self.assertEqual(model.report['recommended_model'], 'linear')
        self.assertLess(model.report['models']['linear']['test']['rmse'], 1e-5)
        train_rows = model.report['split']['parts']['train']['source_rows']
        baseline = TabularModel.load(self.root/'trained/model-baseline.joblib')
        np.testing.assert_allclose(baseline.predict(frame), frame.iloc[train_rows].y.mean())
        linear = TabularModel.load(self.root/'trained/model-linear.joblib')
        np.testing.assert_allclose(linear.predict(frame), frame.y, atol=1e-5)
        for name in ('comparison.csv', 'comparison_predictions.csv', 'comparison.html'):
            self.assertTrue((self.root/'trained'/name).is_file())

    def test_test_targets_do_not_change_selection_or_weights(self):
        frame = pd.DataFrame({'x': np.arange(40), 'y': np.arange(40)*3+5})
        config = {'target': 'y', 'seed': 19}
        model, _ = train_table(frame, config, FAST)
        changed = frame.copy()
        test_rows = model.report['split']['parts']['test']['source_rows']
        changed.loc[test_rows, 'y'] = 999999
        other, _ = train_table(changed, config, FAST)
        self.assertEqual(model.report['recommended_model'], other.report['recommended_model'])
        for kind in model.checkpoints:
            np.testing.assert_array_equal(model.predict(frame, kind), other.predict(frame, kind))
        self.assertNotEqual(model.report['models']['linear']['test'], other.report['models']['linear']['test'])

    def test_four_service_models_and_same_recursive_protocol(self):
        data = self.root/'prepared'
        prepare(ROOT/'examples/raw', data, '2027-01-01')
        demand = compare_service('materials', data, self.root/'demand', settings=FAST)
        price = compare_service('price', data, self.root/'price', settings=FAST)
        self.assertEqual(set(demand['models']), {'mlp', 'linear', 'last_value', 'mean3'})
        self.assertEqual(set(price['models']), {'mlp', 'linear', 'baseline', 'formula'})
        self.assertLess(price['models']['formula']['test']['mae'], .05)
        self.assertEqual(price['units'], 'currency')
        for folder, prefix in [('demand', 'material'), ('price', 'price')]:
            for kind in ('mlp', 'linear'):
                checkpoint = self.root/folder/f'{prefix}-{kind}.pth'
                self.assertTrue(checkpoint.exists())
                result = run('materials' if prefix == 'material' else 'price', data,
                             self.root/f'{folder}-{kind}-reload', train=False, model=checkpoint, months=2)
                self.assertTrue(result)
        mlp = torch.load(self.root/'demand/material-mlp.pth', weights_only=False)
        linear = torch.load(self.root/'demand/material-linear.pth', weights_only=False)
        self.assertEqual(mlp['feature_mean'], linear['feature_mean'])
        self.assertEqual(mlp['feature_std'], linear['feature_std'])
        self.assertEqual(mlp['metrics']['split'], linear['metrics']['split'])
        rows = pd.read_csv(self.root/'demand/comparison_predictions.csv')
        expected = rows[rows.model == 'mlp'][['service_id', 'year', 'month', 'lead', 'actual']].reset_index(drop=True)
        for kind in ('linear', 'last_value', 'mean3'):
            pd.testing.assert_frame_equal(expected, rows[rows.model == kind][expected.columns].reset_index(drop=True))

    def test_invalid_stopping_parameters(self):
        for values in ({'patience': 0}, {'min_delta': -1}, {'seed': -1}):
            with self.assertRaises(ValueError):
                TrainingConfig(epochs=50, **values)
