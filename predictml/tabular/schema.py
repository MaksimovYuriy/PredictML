from dataclasses import dataclass, field
import numpy as np
import pandas as pd


def numeric(series, column, *, allow_missing=True):
    raw = series.replace(r'^\s*$', np.nan, regex=True)
    values = pd.to_numeric(raw, errors='coerce')
    invalid = (raw.notna() & values.isna()) | (values.notna() & ~np.isfinite(values))
    if not allow_missing:
        invalid |= values.isna()
    if invalid.any():
        rows = (np.flatnonzero(invalid.to_numpy())[:5]+2).tolist()
        raise ValueError(f'{column}: expected finite numbers at table rows {rows}')
    return values.astype(float)


def dates(series, column):
    raw = series.replace(r'^\s*$', np.nan, regex=True)
    try:
        values = pd.to_datetime(raw, errors='raise', format='ISO8601')
        if values.dt.tz is not None:
            raise ValueError('timezone not supported')
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'{column}: expected ISO dates without timezone') from exc
    return values


@dataclass(frozen=True)
class TableConfig:
    target: str
    features: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    types: dict = field(default_factory=dict)
    seed: int = 42
    temporal: dict | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('configuration: expected JSON object')
        unknown = set(value)-{'target', 'features', 'exclude', 'types', 'seed', 'temporal'}
        if unknown:
            raise ValueError(f'unknown configuration fields: {sorted(unknown)}')
        if 'target' not in value:
            raise ValueError('target: required')
        data = dict(value)
        for key in ('features', 'exclude'):
            if data.get(key) is not None:
                if not isinstance(data[key], (list, tuple)):
                    raise ValueError(f'{key}: expected column list')
                if any(not isinstance(c, str) for c in data[key]):
                    raise ValueError(f'{key}: expected column names')
                data[key] = tuple(data[key])
        if data.get('exclude', ()) is None:
            raise ValueError('exclude: expected column list')
        return cls(**data)

    def resolve(self, frame):
        if not isinstance(self.target, str) or self.target not in frame:
            raise ValueError(f'target: missing column {self.target!r}')
        if not isinstance(self.types, dict) or any(t not in ('numeric', 'categorical', 'datetime') for t in self.types.values()):
            raise ValueError('types: expected numeric/categorical/datetime')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed: expected integer in [0, 2**32)')
        if len(set(self.exclude)) != len(self.exclude) or set(self.exclude)-set(frame):
            raise ValueError('exclude: duplicate or unknown columns')
        features = tuple(c for c in frame if c != self.target and c not in self.exclude) if self.features is None else tuple(self.features)
        if not features or len(set(features)) != len(features):
            raise ValueError('features: select nonempty unique columns')
        if set(features)-set(frame):
            raise ValueError(f'features: missing columns {sorted(set(features)-set(frame))}')
        if self.target in features or set(features) & set(self.exclude):
            raise ValueError('features: target and excluded columns cannot be selected')
        if set(self.types)-set(features):
            raise ValueError('types: overrides must refer to selected features')
        kinds = {}
        for name in features:
            if name in self.types:
                kinds[name] = self.types[name]
                continue
            raw = frame[name].replace(r'^\s*$', np.nan, regex=True).dropna()
            if raw.empty:
                raise ValueError(f'{name}: all values missing; specify type explicitly')
            try:
                numeric(raw, name)
                kinds[name] = 'numeric'
            except ValueError:
                kinds[name] = 'categorical'
        return features, kinds
