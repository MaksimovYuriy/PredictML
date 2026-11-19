from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from predictml.config import TrainingConfig
from predictml.training import train_table
from predictml.workflow import train_file
from predictml.prediction import TabularModel
from predictml.data import read_table, preview, sheets

ROOT = Path(__file__).resolve().parents[1]
FAST = TrainingConfig(epochs=2, validation_interval=1)


def data():
    return pd.DataFrame({'id': range(30), 'area': [float(i+10) for i in range(30)],
                         'district': ['north', 'south']*15,
                         'built_at': pd.date_range('2000-01-01', periods=30).strftime('%Y-%m-%d'),
                         'price': [float(i*2+100) for i in range(30)]})


CONFIG = dict(target='price', features=['area', 'district', 'built_at'], exclude=['id'],
              types={'area': 'numeric', 'district': 'categorical', 'built_at': 'datetime'}, seed=7)


class TabularTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_train_only_statistics_unknown_categories_and_reordered_columns(self):
        frame = data()
        parts = np.random.default_rng(7).permutation(len(frame))
        train, heldout = parts[:18], parts[18:]
        frame.loc[train[0], 'area'] = np.nan
        frame.loc[heldout, 'area'] = 100000
        frame.loc[heldout, 'district'] = 'unseen'
        model, test = train_table(frame, CONFIG, FAST)
        transformer = model.preprocessor.pipeline
        imputer = transformer.named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(imputer.statistics_[0], frame.iloc[train].area.median())
        encoder = transformer.named_transformers_['categorical'].named_steps['encode']
        self.assertNotIn('value:unseen', encoder.categories_[0])
        before = imputer.statistics_.copy()
        prediction = model.predict(frame)
        reordered = frame[['district', 'price', 'id', 'built_at', 'area']]
        np.testing.assert_array_equal(prediction, model.predict(reordered))
        np.testing.assert_array_equal(before, imputer.statistics_)
        transformed = model.preprocessor.transform(frame.iloc[heldout])
        cat_slice = transformer.output_indices_['categorical']
        np.testing.assert_array_equal(transformed[:, cat_slice], 0)
        self.assertEqual(model.checkpoint['input_size'], transformed.shape[1])
        self.assertEqual(len(test), 6)

    def test_missing_targets_removed_and_invalid_target_rejected(self):
        frame = data()
        frame.loc[0, 'price'] = np.nan
        model, _ = train_table(frame, CONFIG, FAST)
        self.assertEqual(model.report['dropped_missing_target'], 1)
        for part in model.report['split']['parts'].values():
            self.assertNotIn(0, part['source_rows'])
        frame['price'] = frame['price'].astype(object)
        frame.loc[1, 'price'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'price: expected finite'):
            train_table(frame, CONFIG, FAST)

    def test_schema_rejects_missing_invalid_numeric_and_dates(self):
        model, _ = train_table(data(), CONFIG, FAST)
        with self.assertRaisesRegex(ValueError, 'missing feature.*area'):
            model.predict(data().drop(columns='area'))
        changed = data()
        changed['area'] = changed.area.astype(object)
        changed.loc[0, 'area'] = 'large'
        with self.assertRaisesRegex(ValueError, 'area: expected finite'):
            model.predict(changed)
        changed = data()
        changed.loc[0, 'built_at'] = 'yesterday'
        with self.assertRaisesRegex(ValueError, 'built_at: expected ISO'):
            model.predict(changed)
        with self.assertRaisesRegex(ValueError, 'target.*excluded'):
            train_table(data(), {**CONFIG, 'features': ['price']}, FAST)

    def test_all_missing_and_second_schema(self):
        frame = pd.DataFrame({'a': range(20), 'empty': [np.nan]*20, 'category': [None]*20,
                              'result': np.arange(20)*.5})
        model, _ = train_table(frame, dict(target='result', types={'empty': 'numeric', 'category': 'categorical'}), FAST)
        self.assertEqual(model.schema['features'], ['a', 'empty', 'category'])
        self.assertTrue(np.isfinite(model.predict(frame)).all())
        self.assertEqual(model.checkpoint['input_size'], 3)
        with self.assertRaisesRegex(ValueError, 'all values missing'):
            train_table(frame, dict(target='result'), FAST)

    def test_csv_xlsx_sheet_selection_preview_and_duplicates(self):
        path = self.root/'tables.xlsx'
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            pd.DataFrame({'note': ['other sheet']}).to_excel(writer, sheet_name='notes', index=False)
            data().to_excel(writer, sheet_name='training', index=False)
        self.assertEqual(sheets(path), ['notes', 'training'])
        self.assertEqual(len(read_table(path, 'training')), 30)
        self.assertEqual(len(preview(path, 'training', 3)['preview']), 3)
        with self.assertRaisesRegex(ValueError, 'unknown sheet'):
            read_table(path, 'missing')
        csv = self.root/'bad.csv'
        csv.write_text('a,a\n1,2\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_table(csv)
        with pd.ExcelWriter(self.root/'bad.xlsx', engine='openpyxl') as writer:
            pd.DataFrame([['a', 'a'], [1, 2]]).to_excel(writer, header=False, index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_table(self.root/'bad.xlsx')

    def test_bundle_and_prediction_after_restart(self):
        csv = self.root/'training.csv'
        data().to_csv(csv, index=False)
        model = train_file(csv, self.root/'model', CONFIG, settings=FAST)
        source = self.root/'input.xlsx'
        data().drop(columns='price').to_excel(source, sheet_name='input', index=False)
        output = self.root/'predictions.xlsx'
        subprocess.run([sys.executable, '-m', 'predictml', 'predict-table', '--file', str(source),
                        '--sheet', 'input', '--model', str(self.root/'model/model.joblib'), '--output', str(output)],
                       cwd=ROOT, capture_output=True, check=True, text=True)
        np.testing.assert_allclose(pd.read_excel(output).prediction, model.predict(data()), rtol=1e-7)
        loaded = TabularModel.load(self.root/'model/model.joblib')
        self.assertEqual(loaded.schema, model.schema)
        self.assertEqual(loaded.preprocessor.output_names, model.preprocessor.output_names)

    def test_temporal_split_and_purge(self):
        frame = pd.DataFrame({'origin': pd.date_range('2023-01-01', periods=30, freq='MS').strftime('%Y-%m-%d'),
                              'x': np.arange(30), 'future_y': np.arange(30)+2})
        config = dict(target='future_y', features=['x'], temporal=dict(time_column='origin', frequency='MS',
                      horizon=1, target_definition='next month total, prepared by source', available_features=['x']))
        model, _ = train_table(frame, config, FAST)
        split = model.report['split']
        self.assertEqual(split['purged_rows'], 2)
        self.assertEqual(split['parts']['train']['source_rows'], list(range(17)))
        self.assertEqual(split['parts']['validation']['source_rows'], list(range(18, 23)))
        self.assertEqual(split['parts']['test']['source_rows'], list(range(24, 30)))
        with self.assertRaisesRegex(ValueError, 'available_features'):
            train_table(frame, {**config, 'temporal': {**config['temporal'], 'available_features': []}}, FAST)
        with self.assertRaisesRegex(ValueError, 'specify'):
            train_table(frame, {**config, 'temporal': {'time_column': 'origin'}}, FAST)

    def test_automatic_types_are_inferred_on_train_only(self):
        frame = pd.DataFrame({'x': np.arange(30).astype(object), 'y': np.arange(30)})
        heldout = np.random.default_rng(42).permutation(30)[18:]
        frame.loc[heldout, 'x'] = 'not a number'
        with self.assertRaisesRegex(ValueError, 'x: expected finite'):
            train_table(frame, dict(target='y'), FAST)
