"""Apply a fitted table schema, preprocessor and selected checkpoint."""
import numpy as np
import torch
from .models import Predictor


class TabularModel:
    format_version = 1

    def __init__(self, schema, preprocessor, checkpoint, report):
        self.format_version = 1
        self.schema, self.preprocessor = schema, preprocessor
        self.checkpoint, self.report = checkpoint, report

    def predict(self, frame, model_kind=None):
        if frame.empty:
            raise ValueError('prediction: no rows')
        values = self.preprocessor.transform(frame)
        with torch.random.fork_rng(devices=[]):
            state = self.checkpoint if model_kind is None else getattr(self, 'checkpoints', {'mlp': self.checkpoint})[model_kind]
            result = Predictor(state).predict(values).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError('prediction: non-finite model output')
        return result

    def save(self, path):
        from .storage.bundles import save_table_model
        save_table_model(self, path)

    @classmethod
    def load(cls, path):
        from .storage.bundles import load_table_model
        return load_table_model(path)
