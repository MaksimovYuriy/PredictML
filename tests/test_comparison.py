from predictml.demo import examples_dir
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from predictml.config import TrainingConfig
from predictml.training import train_table
from predictml.workflow import train_file
from predictml.prediction import TabularModel
from predictml.workflow import compare_service
from predictml.workflow import run
from predictml.scenarios.serviceflow import prepare

FAST = TrainingConfig(epochs=4, validation_interval=1, patience=2, seed=19)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

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
        prepare(examples_dir()/'raw', data, '2027-01-01')
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

