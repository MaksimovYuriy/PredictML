"""Run after reconstructing and installing the application from module 2."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from predictml.config import TrainingConfig
from predictml.demo import export_examples
from predictml.evaluation import regression_metrics
from predictml.scenarios.materials import calculate
from predictml.storage import Store
from predictml.workflow import train_file, update_version, compare_versions


class ProgramChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_metrics_and_missing_denominator(self):
        self.assertAlmostEqual(regression_metrics([1, 2, 3], [2, 2, 2])['mae'], 2/3)
        self.assertIsNone(regression_metrics([0, 0], [1, 1])['wmape'])

    def test_training_and_loading_in_new_process(self):
        source = self.root/'train.csv'
        pd.DataFrame({'x': np.arange(30), 'y': np.arange(30)*3+5}).to_csv(source, index=False)
        store = self.root/'store'
        model = train_file(source, store, {'target': 'y', 'seed': 42},
                           settings=TrainingConfig(epochs=2, validation_interval=1, seed=42),
                           model_kind='linear')
        self.assertEqual(model.report['recommended_model'], 'linear')
        self.assertLess(model.report['test']['mae'], 1e-4)
        self.assertEqual(len(Store(store).versions()), 3)
        destination = self.root/'predictions.csv'
        subprocess.run([sys.executable, '-m', 'predictml', 'predict-table',
                        '--file', str(source), '--model', str(store/'model-linear.joblib'),
                        '--output', str(destination)], check=True)
        np.testing.assert_allclose(pd.read_csv(destination).prediction,
                                   np.arange(30)*3+5, atol=1e-4)

    def test_decimal_purchase(self):
        materials = [dict(id=1, title='Liquid', unit='ml', purchase_step=.25,
                          current_quantity=.2, minimal_quantity=.1, price=2.675)]
        norms = [dict(service_id=1, material_id=1, required_quantity=.1),
                 dict(service_id=2, material_id=1, required_quantity=.2)]
        demand = [dict(service_id=s, year=2027, month=m, predicted_bookings=1)
                  for s in (1, 2) for m in (1, 2, 3)]
        result = calculate(demand, materials, norms)[0]
        self.assertEqual(result['purchase_qty'], 1)
        self.assertEqual(result['purchase_cost'], 2.68)

    def test_updates_and_independent_control(self):
        examples = export_examples(self.root/'examples')
        config = json.loads((examples/'tabular/housing-config.json').read_text())
        root = self.root/'store'
        settings = TrainingConfig(epochs=2, validation_interval=1, seed=42)
        train_file(examples/'tabular/housing.csv', root, config, settings=settings)
        store = Store(root)
        parent = next(r['id'] for r in store.versions() if r['model_kind'] == 'mlp')
        child = update_version(root, parent, examples/'tabular/housing_update.csv',
                               operation='fine-tune', settings=settings)
        report = compare_versions(root, [parent, child['version_id']],
                                  examples/'tabular/housing_control.csv')
        self.assertEqual(report['control_rows'], 10)
        with self.assertRaises(ValueError):
            compare_versions(root, [parent, child['version_id']], examples/'tabular/housing.csv')
        self.assertEqual(len(store.versions()), 4)

    def test_user_interface_without_browser_server(self):
        import predictml
        from streamlit.testing.v1 import AppTest
        env = {'PREDICTML_WORKSPACE': str(self.root/'work'),
               'PREDICTML_STORE': str(self.root/'models')}
        with patch.dict(os.environ, env):
            app = AppTest.from_file(Path(predictml.__file__).parent/'ui/app.py',
                                    default_timeout=60).run()
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))


if __name__ == '__main__':
    unittest.main(verbosity=2)
