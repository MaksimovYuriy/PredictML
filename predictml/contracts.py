"""Validation of the raw ServiceFlow export boundary."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

RAW_COLUMNS = {
    'notes': 'service_id client_id status start_at total_price'.split(),
    'services': 'id price active'.split(),
    'materials': 'id title quantity minimal_quantity price'.split(),
    'service_materials': 'service_id material_id required_quantity'.split(),
}


def read_csv(path, columns):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError(f'{path.name}: duplicate column names')
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f'{path.name}: missing header') from exc
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    return table


def number(table, name, column, *, integer=False, positive=False, nullable=False):
    raw = table[column]
    values = pd.to_numeric(raw, errors='coerce')
    invalid = (~np.isfinite(values) | values.isna())
    if nullable:
        invalid &= raw.notna()
    invalid |= values < (1 if positive and integer else 0)
    if positive:
        invalid |= values <= 0
    if integer:
        invalid |= values.notna() & (values % 1 != 0)
    if invalid.any():
        rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
        raise ValueError(f'{name}.{column}: invalid numeric value at CSV rows {rows}')
    table[column] = values


def load_raw(data_dir):
    tables = {name: read_csv(Path(data_dir)/f'{name}.csv', cols)
              for name, cols in RAW_COLUMNS.items()}
    for name, table in tables.items():
        for column in ('id', 'service_id', 'client_id', 'material_id'):
            if column in table:
                number(table, name, column, integer=True, positive=True)
        if 'id' in table and table.id.duplicated().any():
            raise ValueError(f'{name}.id: duplicate identifiers')
    if tables['services'].empty:
        raise ValueError('services: no services')
    for name, columns in {'notes': ['total_price'], 'services': ['price'],
                          'materials': ['quantity', 'minimal_quantity', 'price'],
                          'service_materials': ['required_quantity']}.items():
        for column in columns:
            number(tables[name], name, column, integer=column in ('quantity', 'minimal_quantity', 'required_quantity'),
                   positive=name == 'services', nullable=name == 'materials' and column == 'price')
    for name in ('notes', 'service_materials'):
        if not tables[name].service_id.isin(tables['services'].id).all():
            raise ValueError(f'{name}.service_id: unknown service identifier')
    if not tables['service_materials'].material_id.isin(tables['materials'].id).all():
        raise ValueError('service_materials.material_id: unknown material identifier')
    titles = tables['materials'].title
    if titles.isna().any() or titles.astype(str).str.strip().eq('').any():
        raise ValueError('materials.title: non-empty title required')
    active = tables['services'].active.astype(str).str.lower()
    if not active.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
        raise ValueError('services.active: expected true/false or 1/0')
    tables['services']['active'] = active.isin(['true', '1', '1.0'])
    statuses = tables['notes'].status.astype(str)
    mapping = {'0': 0, '1': 1, '2': 2, 'pending': 0, 'canceled': 1, 'completed': 2}
    if not statuses.isin(mapping).all():
        raise ValueError('notes.status: expected 0/1/2 or pending/canceled/completed')
    tables['notes']['status'] = statuses.map(mapping)
    try:
        dates = pd.to_datetime(tables['notes'].start_at, errors='raise', format='ISO8601')
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError('missing date or timezone')
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError('notes.start_at: expected valid timezone-free calendar timestamps') from exc
    tables['notes']['start_at'] = dates
    return tables

MATERIAL_COLUMNS = 'service_id total_bookings avg_bookings booking_ratio bookings_lag_1 bookings_lag_2 bookings_lag_3 avg_bookings_last_3m usage_trend month_sin month_cos year month'.split()
PRICE_COLUMNS = 'service_id base_price is_active total_bookings revenue avg_price_fact first_visit_share year month avg_material_cost_per_service material_cost_share target_price'.split()


def validate_prepared(data_dir, scenario, train):
    """Reject malformed prepared exports before importing/running PyTorch."""
    import json
    data = Path(data_dir)
    material = scenario == 'materials'
    name = 'material_forecast' if material else 'price_analysis'
    table = read_csv(data/f'{name}.csv', MATERIAL_COLUMNS if material else PRICE_COLUMNS)
    if table.empty:
        raise ValueError(f'{name}: no rows')
    for column in (MATERIAL_COLUMNS if material else PRICE_COLUMNS):
        if column in ('month_sin', 'month_cos'):
            values = pd.to_numeric(table[column], errors='coerce')
            if not np.isfinite(values).all() or (values.abs() > 1).any():
                raise ValueError(f'{name}.{column}: expected finite value in [-1, 1]')
        else:
            number(table, name, column,
                   integer=column in ('service_id', 'year', 'month', 'total_bookings', 'is_active'),
                   positive=column in ('service_id', 'year', 'month', 'avg_bookings', 'base_price'))
    if (table.month > 12).any() or (table.year > 9999).any():
        raise ValueError(f'{name}: invalid year/month')
    if table.duplicated(['service_id', 'year', 'month']).any():
        raise ValueError(f'{name}: duplicate service_id/year/month')
    if not material:
        if not table.is_active.isin([0, 1]).all() or (table.first_visit_share > 1).any():
            raise ValueError(f'{name}: invalid is_active or first_visit_share')
    if train:
        ym = table.year * 12 + table.month
        if not material:
            ym = ym[table.year < table.year.max()]
            if ym.nunique() < 7:
                raise ValueError(f'{name}: need at least 7 training months before the prediction year')
        else:
            if not ((ym < ym.max()-5).any() and
                    ((ym >= ym.max()-5) & (ym < ym.max()-2)).any() and
                    (ym >= ym.max()-2).any()):
                raise ValueError(f'{name}: empty train/validation/test period; extend history')
    if material:
        records = {}
        for filename, columns in {
            'materials': ['id', 'title', 'current_quantity', 'minimal_quantity', 'price'],
            'service_materials': ['service_id', 'material_id', 'required_quantity'],
        }.items():
            try:
                values = json.loads((data/f'{filename}.json').read_text(encoding='utf-8'))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{filename}.json: invalid JSON') from exc
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ValueError(f'{filename}.json: expected array of objects')
            frame = pd.DataFrame(values, columns=columns) if not values else pd.DataFrame(values)
            if set(columns) - set(frame.columns):
                raise ValueError(f'{filename}.json: missing required fields')
            for column in columns:
                if column == 'title':
                    if frame.title.isna().any() or frame.title.astype(str).str.strip().eq('').any():
                        raise ValueError('materials.title: non-empty title required')
                else:
                    number(frame, filename, column, integer=column != 'price',
                           positive=column in ('id', 'service_id', 'material_id'))
            # JSON is consumed directly by the legacy predictor: do not accept
            # numeric strings that validation would convert only in this frame.
            for record in values:
                for column in columns:
                    if column != 'title' and (isinstance(record[column], bool) or not isinstance(record[column], (int, float))):
                        raise ValueError(f'{filename}.{column}: expected JSON number')
            records[filename] = frame
        mats, norms = records['materials'], records['service_materials']
        if mats.id.duplicated().any():
            raise ValueError('materials.id: duplicate identifiers')
        if not norms.material_id.isin(mats.id).all():
            raise ValueError('service_materials.material_id: unknown material identifier')
        # Services with zero historical bookings are omitted by the original
        # Ruby exporter; their norms legitimately remain in the JSON.
        if table.groupby('service_id').avg_bookings.nunique().gt(1).any():
            raise ValueError('material_forecast.avg_bookings: inconsistent value within service')
    return table
