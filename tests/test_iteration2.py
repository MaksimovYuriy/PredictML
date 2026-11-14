"""Compatibility tests against unchanged ServiceFlow Python snapshots."""
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from importlib.util import spec_from_file_location, module_from_spec
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from predictml.config import CONFIGS, TrainingConfig
from predictml.models import MLP
from predictml.pipeline import run
from predictml.preparation import prepare

ROOT = Path(__file__).resolve().parents[1]


def reference(name):
    spec = spec_from_file_location(name, ROOT/'tests/reference/python'/f'{name}.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'data'
        prepare(ROOT/'examples/raw', self.data, '2026-12-31')
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, state)

    def test_training_weights_metrics_and_predictions_match_reference(self):
        for scenario in ('materials', 'price'):
            with self.subTest(scenario=scenario):
                prefix = 'material' if scenario == 'materials' else 'price'
                csv = self.data/('material_forecast.csv' if scenario == 'materials' else 'price_analysis.csv')
                old = reference(f'train_{scenario}')
                old.CSV_PATH = str(csv)
                old.MODEL_PATH = str(self.root/f'{prefix}-original.pth')
                old.METRICS_PATH = str(self.root/f'{prefix}-original-metrics.json')
                old.EPOCHS = 50
                with redirect_stdout(io.StringIO()):
                    torch.manual_seed(31)
                    old.train()
                    torch.manual_seed(31)
                    actual = run(scenario, self.data, self.root/scenario, epochs=50)
                expected_state = torch.load(old.MODEL_PATH, weights_only=False)
                actual_state = torch.load(self.root/scenario/f'{prefix}_model.pth', weights_only=False)
                self.assertEqual(expected_state.keys(), actual_state.keys())
                for key in expected_state:
                    if key == 'model_state':
                        for parameter, value in expected_state[key].items():
                            torch.testing.assert_close(value, actual_state[key][parameter], rtol=0, atol=0)
                    else:
                        self.assertEqual(expected_state[key], actual_state[key])
                old_predict = reference(f'predict_{scenario}')
                old_predict.CSV_PATH = str(csv)
                old_predict.MODEL_PATH = old.MODEL_PATH
                old_predict.OUTPUT_PATH = str(self.root/f'{prefix}-original-predictions.json')
                if scenario == 'materials':
                    old_predict.SERVICE_MATERIALS_PATH = str(self.data/'service_materials.json')
                    old_predict.MATERIALS_PATH = str(self.data/'materials.json')
                with redirect_stdout(io.StringIO()):
                    old_predict.predict()
                self.assertEqual(actual, json.loads(Path(old_predict.OUTPUT_PATH).read_text()))
                reloaded = run(scenario, self.data, self.root/f'{scenario}-reloaded',
                               train=False, model=old.MODEL_PATH)
                self.assertEqual(actual, reloaded)

    def test_forward_and_checkpoint_schema_compatibility(self):
        for scenario in ('materials', 'price'):
            config = CONFIGS[scenario]
            original = reference(f'train_{scenario}')
            cls = original.MaterialMLP if scenario == 'materials' else original.PriceMLP
            model = cls(len(config.features), list(config.hidden_sizes))
            shared = MLP(len(config.features), config.hidden_sizes)
            shared.load_state_dict(model.state_dict())
            model.eval()
            shared.eval()
            values = torch.arange(3*len(config.features), dtype=torch.float32).reshape(3, -1)
            torch.testing.assert_close(model(values), shared(values), rtol=0, atol=0)

    def constant_checkpoint(self, scenario, output, ratio):
        config = CONFIGS[scenario]
        model = MLP(len(config.features), config.hidden_sizes)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias.fill_(ratio)
        state = dict(model_state=model.state_dict(), input_size=len(config.features),
                     features=list(config.features), hidden_sizes=list(config.hidden_sizes),
                     feature_mean=[0]*len(config.features), feature_std=[1]*len(config.features))
        torch.save(state, output)
        return output

    def test_concurrent_independent_predictions(self):
        a = self.constant_checkpoint('materials', self.root/'a.pth', 0.5)
        b = self.constant_checkpoint('materials', self.root/'b.pth', 1.5)
        c = self.constant_checkpoint('price', self.root/'c.pth', 1.1)
        jobs = [('materials', a, 1), ('materials', b, 3), ('price', c, 3)]
        expected = [run(s, self.data, self.root/f'serial-{i}', train=False, model=p, months=h)
                    for i, (s, p, h) in enumerate(jobs)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run, s, self.data, self.root/f'parallel-{i}',
                                   train=False, model=p, months=h)
                       for i, (s, p, h) in enumerate(jobs)]
            self.assertEqual([f.result() for f in futures], expected)
        self.assertNotEqual(expected[0], expected[1])

    def test_checkpoint_rejects_wrong_feature_order(self):
        path = self.constant_checkpoint('materials', self.root/'bad.pth', 1)
        state = torch.load(path, weights_only=False)
        state['features'].reverse()
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'incompatible feature schema'):
            run('materials', self.data, self.root/'output', train=False, model=path)

    def test_training_settings_validation(self):
        for values in ({'epochs': 49}, {'epochs': 50, 'batch_size': 0},
                       {'epochs': 50, 'learning_rate': float('nan')}):
            with self.assertRaises(ValueError):
                TrainingConfig(**values)
