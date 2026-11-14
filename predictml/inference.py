"""Load and apply a model without scenario-specific globals."""
from pathlib import Path
import numpy as np
import torch
from .models import MLP


class Predictor:
    def __init__(self, checkpoint):
        self.demand_context = checkpoint.get('demand_context')
        self.features = tuple(checkpoint['features'])
        self.mean = np.array(checkpoint['feature_mean'], dtype=np.float32)
        self.std = np.array(checkpoint['feature_std'], dtype=np.float32)
        size = checkpoint['input_size']
        if len(self.features) != size or self.mean.shape != (size,) or self.std.shape != (size,):
            raise ValueError('checkpoint: inconsistent feature dimensions')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('checkpoint: invalid normalization statistics')
        if len(set(self.features)) != len(self.features):
            raise ValueError('checkpoint: duplicate feature names')
        self.kind = checkpoint.get('model_kind', 'mlp')
        if self.kind in ('linear', 'baseline'):
            self.coefficients = np.asarray(checkpoint['coefficients'], dtype=float)
            self.intercept = float(checkpoint['intercept'])
            if self.coefficients.shape != (size,) or not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercept):
                raise ValueError('checkpoint: invalid linear coefficients')
            return
        if self.kind != 'mlp':
            raise ValueError('checkpoint: unsupported model kind')
        self.model = MLP(size, checkpoint['hidden_sizes'], checkpoint.get('dropout', 0.1))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    @classmethod
    def load(cls, path):
        # Original trusted checkpoint format; portable CPU loading.
        return cls(torch.load(Path(path), map_location='cpu', weights_only=False))

    def predict(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError('prediction: incompatible feature dimensions')
        if not np.isfinite(values).all():
            raise ValueError('prediction: non-finite feature values')
        if self.kind in ('linear', 'baseline'):
            return (((values-self.mean)/self.std) @ self.coefficients+self.intercept).reshape(-1, 1)
        with torch.no_grad():
            return self.model(torch.tensor((values-self.mean)/self.std)).numpy()
