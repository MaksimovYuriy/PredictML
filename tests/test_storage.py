from predictml.demo import examples_dir
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from predictml.config import TrainingConfig, PRICE
from predictml.workflow import train_file
from predictml.workflow import compare_service
from predictml.scenarios.serviceflow import prepare
from predictml.models import Predictor
from predictml.storage import Store, execute_run, verify_bundle, load_payload
from predictml.storage.bundles import digest
from predictml.workflow import predict_version, export_metrics
from predictml.storage.bundles import checkpoint_spec

ROOT = Path(__file__).resolve().parents[1]
FAST = TrainingConfig(epochs=2, validation_interval=1, seed=42)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def train(self):
        source = self.root/'training.csv'
        pd.DataFrame({'x': np.arange(20), 'y': 2*np.arange(20)+3}).to_csv(source, index=False)
        return train_file(source, self.root/'store', {'target': 'y'}, settings=FAST)

    def test_new_training_preserves_previous_versions(self):
        first = self.train()
        store = Store(self.root/'store')
        old = {row['id']: (store.version(row['id'])[1], digest(store.version(row['id'])[1]/'manifest.json')) for row in store.versions()}
        before = {key: load_payload(path)[1].predict(pd.DataFrame({'x': [5, 10]})) for key, (path, _) in old.items()}
        second = self.train()
        self.assertNotEqual(first.storage['run_id'], second.storage['run_id'])
        self.assertEqual(len(store.versions()), 6)
        self.assertEqual([r['status'] for r in store.runs()], ['completed', 'completed'])
        for key, (path, checksum) in old.items():
            self.assertEqual(digest(path/'manifest.json'), checksum)
            np.testing.assert_array_equal(store.load(key)[2].predict(pd.DataFrame({'x': [5, 10]})), before[key])

    def test_hash_before_deserialization_and_missing_file(self):
        self.train()
        store = Store(self.root/'store')
        version = store.versions()[0]['id']
        _, directory, _ = store.version(version)
        with (directory/'model.joblib').open('ab') as stream:
            stream.write(b'corrupt')
        with patch('joblib.load', side_effect=AssertionError('must not deserialize')):
            with self.assertRaisesRegex(ValueError, 'checksum/size'):
                store.load(version)
        (directory/'model.joblib').unlink()
        with self.assertRaisesRegex(ValueError, 'missing payload'):
            store.load(version)

    def test_unsupported_and_incompatible_metadata(self):
        self.train()
        store = Store(self.root/'store')
        directory = store.version(store.versions()[0]['id'])[1]
        copy = self.root/'portable'
        shutil.copytree(directory, copy)
        path = copy/'manifest.json'
        original = json.loads(path.read_text())
        changed = json.loads(json.dumps(original))
        changed['environment']['packages']['numpy'] = '0.0.0'
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'incompatible dependency numpy'):
            load_payload(copy)
        changed = dict(original)
        changed.pop('metrics')
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'incomplete manifest'):
            verify_bundle(copy)
        changed = dict(original, bundle_format=999)
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            verify_bundle(copy)
        path.write_text(json.dumps(original))
        manifest, model = load_payload(copy)
        self.assertEqual(manifest['version_id'], original['version_id'])
        self.assertEqual(len(model.predict(pd.DataFrame({'x': [1]}))), 1)

    def test_failed_worker_and_failed_commit_publish_no_versions(self):
        root = self.root/'failed'
        def fail(stage, run_id):
            (stage/'partial.txt').write_text('partial')
            raise ValueError('deliberate failure')
        with self.assertRaisesRegex(ValueError, 'deliberate'):
            execute_run(root, 'test', {}, fail)
        self.assertEqual(Store(root).runs()[0]['status'], 'failed')
        self.assertEqual(Store(root).versions(), [])
        self.assertEqual(list((root/'runs').iterdir()), [])
        with patch.object(Store, 'complete', side_effect=sqlite3.OperationalError('commit failure')):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'commit failure'):
                execute_run(root, 'test', {}, lambda stage, rid: (stage/'result.txt').write_text('done'))
        self.assertTrue(all(r['status'] == 'failed' for r in Store(root).runs()))
        self.assertEqual(list((root/'runs').iterdir()), [])

    def test_restart_predict_by_id_and_export(self):
        self.train()
        store = Store(self.root/'store')
        version = next(v['id'] for v in store.versions() if v['model_kind'] == 'linear')
        new = self.root/'new.csv'
        pd.DataFrame({'x': [10, 20]}).to_csv(new, index=False)
        (self.root/'training.csv').unlink()
        output = self.root/'prediction'
        subprocess.run([sys.executable, '-m', 'predictml', 'predict-version', '--store', str(store.root),
                        '--id', version, '--data', str(new), '--output', str(output)],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        np.testing.assert_allclose(pd.read_csv(output/'predictions.csv').prediction, [23, 43], atol=1e-5)
        self.assertEqual(pd.ExcelFile(output/'results.xlsx').sheet_names, ['predictions', 'metrics'])
        self.assertEqual(json.loads((output/'prediction.json').read_text())['version_id'], version)
        for suffix in ('csv', 'xlsx'):
            export_metrics(store.root, version, self.root/f'metrics.{suffix}')
        a = pd.read_csv(self.root/'metrics.csv')
        b = pd.read_excel(self.root/'metrics.xlsx')
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
        self.assertEqual(len(store.versions()), 3)
        self.assertEqual([r['operation'] for r in store.runs()][-3:], ['predict-version', 'export-metrics', 'export-metrics'])

    def test_four_checkpoint_versions_and_domain_exports(self):
        data = self.root/'data'
        prepare(examples_dir()/'raw', data, '2027-01-01')
        for scenario in ('materials', 'price'):
            root = self.root/scenario
            compare_service(scenario, data, root, settings=FAST)
            store = Store(root)
            self.assertEqual(len(store.versions()), 2)
            for row in store.versions():
                bundle, manifest, state = store.load(row['id'])
                self.assertEqual(manifest['model_kind'], row['model_kind'])
                self.assertEqual(Predictor.load(bundle).features, tuple(state['features']))
            linear = next(row['id'] for row in store.versions() if row['model_kind'] == 'linear')
            result = predict_version(root, linear, data, self.root/f'{scenario}-export', months=2)
            sheets = pd.ExcelFile(self.root/f'{scenario}-export/results.xlsx').sheet_names
            self.assertIn('metrics', sheets)
            self.assertIn('materials' if scenario == 'materials' else 'prices', sheets)
            self.assertEqual(result['version_id'], linear)

    def test_parallel_runs_are_distinct(self):
        root = self.root/'parallel'
        Store(root)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(execute_run, root, 'test', {'i': i},
                                   lambda stage, rid: (stage/'value.txt').write_text(rid), publish=False) for i in range(2)]
            results = [f.result() for f in futures]
        self.assertEqual(len({r[1] for r in results}), 2)
        self.assertTrue(all(row['status'] == 'completed' for row in Store(root).runs()))

    def test_explicit_interruption_recovery(self):
        store = Store(self.root/'store')
        rid = store.start('interrupted', {})
        other = store.start('active', {})
        store.recover(rid)
        statuses = {r['id']: r['status'] for r in store.runs()}
        self.assertEqual(statuses[rid], 'failed')
        self.assertEqual(statuses[other], 'running')
        with self.assertRaises(ValueError):
            store.recover(rid)

    def test_sql_transaction_rolls_back_partial_version_registration(self):
        root = self.root/'rollback'
        def worker(stage, run_id):
            state = dict(model_kind='linear', input_size=len(PRICE.features), features=list(PRICE.features),
                         feature_mean=[0.]*len(PRICE.features), feature_std=[1.]*len(PRICE.features),
                         coefficients=[0.]*len(PRICE.features), intercept=1.)
            torch.save(state, stage/'price.pth')
        def specs(stage, result):
            first = checkpoint_spec(stage/'price.pth', 'price', {})
            return [first, {**first, 'parent_id': 'unknown-parent'}]
        with self.assertRaises(sqlite3.IntegrityError):
            execute_run(root, 'test', {}, worker, specs)
        self.assertEqual(Store(root).versions(), [])
        self.assertEqual(Store(root).runs()[0]['status'], 'failed')
        self.assertEqual(list((root/'runs').iterdir()), [])

    def test_export_and_training_cannot_overwrite_immutable_bundle(self):
        self.train()
        store = Store(self.root/'store')
        version = store.versions()[0]['id']
        bundle = store.version(version)[1]
        with self.assertRaisesRegex(ValueError, 'immutable run history'):
            export_metrics(store.root, version, bundle/'metrics.csv')
        with self.assertRaisesRegex(ValueError, 'inside a model bundle'):
            Store(bundle)
