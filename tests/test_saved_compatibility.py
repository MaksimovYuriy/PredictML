from predictml.demo import examples_dir
"""Load a real pre-refactor payload and preserve seeded numerical results."""
import json
from pathlib import Path
import unittest
import tempfile
import zlib
import numpy as np
from predictml.config import TrainingConfig
from predictml.data import read_table
from predictml.prediction import TabularModel
from predictml.training import train_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/pre_refactor'


class SavedCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.frame = read_table(examples_dir()/'tabular/housing.csv')
        self.expected = json.loads((FIXTURE/'predictions.json').read_text())

    def assert_predictions(self, model):
        for kind, expected in self.expected.items():
            with self.subTest(kind=kind):
                np.testing.assert_allclose(model.predict(self.frame, kind), expected,
                                           rtol=1e-6, atol=1e-5)

    def test_payload_with_original_class_paths_still_loads(self):
        self.assert_predictions(TabularModel.load(FIXTURE/'table.joblib'))

    def test_seeded_training_preserves_pre_refactor_predictions(self):
        config = json.loads((examples_dir()/'tabular/housing-config.json').read_text())
        model, _ = train_table(self.frame, config,
                               TrainingConfig(epochs=3, validation_interval=1, seed=42))
        self.assert_predictions(model)

    def test_compressed_pre_refactor_payload_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old-compressed.joblib'
            path.write_bytes(zlib.compress((FIXTURE/'table.joblib').read_bytes()))
            self.assert_predictions(TabularModel.load(path))
