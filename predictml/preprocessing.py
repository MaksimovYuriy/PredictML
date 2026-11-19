"""Train-only feature preparation, retained for prediction and continuation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import numeric, dates


class Preprocessor:
    def __init__(self, features, types):
        self.features = tuple(features)
        self.types = dict(types)

    def normalize(self, frame):
        if frame.columns.duplicated().any():
            raise ValueError('duplicate column names')
        missing = set(self.features)-set(frame)
        if missing:
            raise ValueError(f'prediction: missing feature columns {sorted(missing)}')
        output, numbers, categories = {}, [], []
        for i, name in enumerate(self.features):
            kind, key = self.types[name], f'f{i}'
            if kind == 'numeric':
                output[key] = numeric(frame[name], name)
                numbers.append(key)
            elif kind == 'categorical':
                values = frame[name].replace(r'^\s*$', np.nan, regex=True)
                output[key] = values.map(lambda v: 'value:'+str(v) if pd.notna(v) else np.nan).astype(object)
                categories.append(key)
            else:
                value = dates(frame[name], name)
                for suffix, data in [('year', value.dt.year), ('month', value.dt.month),
                                     ('day', value.dt.day), ('weekday', value.dt.dayofweek),
                                     ('hour', value.dt.hour), ('minute', value.dt.minute),
                                     ('month_sin', np.sin(2*np.pi*value.dt.month/12)),
                                     ('month_cos', np.cos(2*np.pi*value.dt.month/12))]:
                    output[f'{key}_{suffix}'] = data.astype(float)
                    numbers.append(f'{key}_{suffix}')
        return pd.DataFrame(output, index=frame.index), numbers, categories

    def fit(self, frame):
        normalized, numbers, categories = self.normalize(frame)
        transforms = []
        if numbers:
            transforms.append(('numeric', Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('scale', StandardScaler())]), numbers))
        if categories:
            transforms.append(('categorical', Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='<missing>', keep_empty_features=True)),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categories))
        self.pipeline = ColumnTransformer(transforms, sparse_threshold=0)
        self.pipeline.fit(normalized)
        self.output_names = tuple(self.pipeline.get_feature_names_out())
        return self

    def transform(self, frame):
        normalized, _, _ = self.normalize(frame)
        values = np.asarray(self.pipeline.transform(normalized), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('preprocessing: non-finite output, check feature magnitudes')
        return values
