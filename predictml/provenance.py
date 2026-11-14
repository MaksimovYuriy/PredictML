"""Conservative observation fingerprints; labels cannot hide reused observations."""
import hashlib
import json
import pandas as pd
from .tabular.schema import numeric, dates


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
