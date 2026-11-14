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
        with torch.no_grad():
            return self.model(torch.tensor((values-self.mean)/self.std)).numpy()
