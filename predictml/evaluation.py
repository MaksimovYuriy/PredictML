"""Regression metrics and independence checks for comparable observations."""
import hashlib
import json
import numpy as np
import pandas as pd
from .data import numeric, dates


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    error = y_true - y_pred
    mse = float((error ** 2).mean())
    mae = float(np.abs(error).mean())
    denominator = float(np.abs(y_true).sum())
    wmape = float(np.abs(error).sum() / denominator * 100) if denominator > 0 else 0.0
    return mse, mae, wmape


def regression_metrics(y_true, y_pred):
    """Original units; undefined relative metrics are represented by None."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if not len(actual) or actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('metrics: expected nonempty finite actual/prediction arrays of equal length')
    error = actual-predicted
    mse = float(np.mean(error**2))
    denominator = float(np.abs(actual).sum())
    variance = float(((actual-actual.mean())**2).sum())
    return dict(mae=float(np.mean(np.abs(error))), mse=mse, rmse=float(np.sqrt(mse)),
                wmape=float(np.abs(error).sum()/denominator*100) if denominator else None,
                r2=float(1-(error**2).sum()/variance) if len(actual) > 1 and variance > 0 else None)


def hashed(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tabular_provenance(frame, schema):
    normalized = {}
    for name in schema['features']:
        kind = schema['types'][name]
        values = numeric(frame[name], name) if kind == 'numeric' else dates(frame[name], name) if kind == 'datetime' else frame[name]
        normalized[name] = [None if pd.isna(v) or str(v).strip() == '' else
                            float(0 if v == 0 else v).hex() if kind == 'numeric' else str(v) for v in values]
    time = (schema.get('temporal') or {}).get('time_column')
    if time and time not in normalized:
        normalized[time] = [str(v) for v in dates(frame[time], time)]
    hashes = [hashed({name: values[i] for name, values in normalized.items()}) for i in range(len(frame))]
    result = dict(policy='feature-observation-v1', complete=True, used_hashes=sorted(set(hashes)), supplied_rows=len(frame))
    if time:
        parsed = dates(frame[time], time)
        temporal = schema['temporal']
        offset = pd.tseries.frequencies.to_offset(temporal['frequency'])*temporal['horizon']
        result.update(time_start=str(parsed.min()), time_end=str(parsed.max()+offset))
    return result


def period_provenance(frame):
    return dict(policy='service-month-v1', complete=True,
                used_hashes=sorted({hashed([int(row.service_id), int(row.year), int(row.month)]) for row in frame.itertuples()}),
                supplied_rows=len(frame), first_period=int((frame.year.astype(int)*12+frame.month.astype(int)).min()),
                last_period=int((frame.year.astype(int)*12+frame.month.astype(int)).max()))


def checkpoint_provenance(directory, task, mode='demand-v2'):
    from pathlib import Path
    name = 'price_analysis.csv' if task == 'price' else 'monthly_demand.csv' if mode == 'demand-v2' else 'material_forecast.csv'
    return period_provenance(pd.read_csv(Path(directory)/name))


def merge_provenance(current, parent, inherit):
    if not inherit:
        return current
    if not parent or not parent.get('complete') or parent.get('policy') != current['policy']:
        return {**current, 'complete': False, 'reason': 'parent training provenance unavailable'}
    result = {**current, 'used_hashes': sorted(set(current['used_hashes']) | set(parent['used_hashes']))}
    if 'last_period' in current:
        result['last_period'] = max(current['last_period'], parent['last_period'])
    if 'time_end' in current:
        result['time_end'] = str(max(pd.Timestamp(current['time_end']), pd.Timestamp(parent['time_end'])))
    return result


def assert_control_independent(control, manifests):
    for manifest in manifests:
        provenance = manifest.get('provenance')
        if not provenance or not provenance.get('complete') or provenance.get('policy') != control['policy']:
            raise ValueError(f"version {manifest['version_id']}: complete compatible training provenance is unavailable; independent comparison is refused")
        if 'time_end' in provenance and pd.Timestamp(control['time_start']) <= pd.Timestamp(provenance['time_end']):
            raise ValueError('temporal control must follow all previously supplied labels')
        if 'last_period' in provenance and control['first_period'] <= provenance['last_period']:
            raise ValueError('period control must follow all previously supplied months')
        if set(provenance['used_hashes']) & set(control['used_hashes']):
            raise ValueError(f"control overlaps data previously supplied for version {manifest['version_id']}; comparison is not independent")


def family(schema):
    return {key: schema.get(key) for key in ('features', 'types', 'target', 'temporal')}


def evaluate_control(manifests, payloads, frame, version_ids, *, history=None):
    """Evaluate already loaded versions on labelled observations; never train or save."""
    task, schema = manifests[0]['task'], manifests[0]['schema']
    if any(m['task'] != task or family(m['schema']) != family(schema) for m in manifests):
        raise ValueError('versions must have the same task, target and input feature schema')
    if task == 'tabular':
        if schema['target'] not in frame:
            raise ValueError('control target column is required')
        actual = numeric(frame[schema['target']], schema['target'], allow_missing=False).to_numpy()
        provenance = tabular_provenance(frame, schema)
        assert_control_independent(provenance, manifests)
        predicted = [model.predict(frame) for model in payloads]
        identities = [dict(source_row=i) for i in range(len(frame))]
        conditions = dict(policy='same independent rows', units='original', schema=schema)
    else:
        if task == 'price':
            from .scenarios.price import control_predictions
        else:
            from .scenarios.demand import control_predictions
        frame, actual, predicted, identities, conditions, provenance, observed = control_predictions(
            frame, payloads, manifests, history=history)
    if not len(frame):
        raise ValueError('control needs labelled observations')
    models, rows = {}, []
    for version, values in zip(version_ids, predicted):
        models[version] = dict(control=regression_metrics(actual, values))
        if task == 'materials':
            models[version]['by_lead'] = {str(lead): regression_metrics(
                [a for a, row in zip(actual, identities) if row['lead'] == lead],
                [p for p, row in zip(values, identities) if row['lead'] == lead]) for lead in range(1, conditions['horizon']+1)}
        rows.extend(dict(**identity, actual=float(a), prediction=float(p), model=version)
                    for identity,a,p in zip(identities,actual,values))
    return frame, models, rows, conditions, provenance, observed if task == 'materials' else None
