"""Validation of the raw ServiceFlow export boundary."""
from pathlib import Path
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
from ..config import (RAW_COLUMNS, MATERIAL_COLUMNS, PRICE_COLUMNS, PRICE_MIN_RATIO,
                      PRICE_MAX_RATIO, PRICE_DEMAND_WEIGHT, PRICE_FIRST_VISIT_WEIGHT)


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


def load_raw(data_dir, *, fractional_quantities=False):
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
            number(tables[name], name, column, integer=not fractional_quantities and column in ('quantity', 'minimal_quantity', 'required_quantity'),
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
            # JSON is consumed directly by the scenario predictor: do not accept
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


def ruby_round(value, digits):
    """Decimal half-away-from-zero for the Ruby export boundary.

    Internal calculations remain floats; this does not promise bitwise parity
    for every possible binary floating-point boundary.
    """
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def prepare_legacy(data_dir, output_dir, as_of, *, tables=None):
    data, output = Path(data_dir), Path(output_dir)
    tables = load_raw(data) if tables is None else tables
    contracts = RAW_COLUMNS
    notes, services, materials, norms = (tables[n] for n in contracts)
    try:
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff) or cutoff.tzinfo is not None:
            raise ValueError('invalid cutoff')
    except (ValueError, TypeError) as exc:
        raise ValueError('as_of: expected valid timezone-free date') from exc
    notes = notes[notes.status.astype(str).isin(['2', 'completed'])].copy()
    if notes.empty:
        raise ValueError('notes: no completed records')
    # PostgreSQL extraction in the source uses database calendar dates.
    # Contract requires timestamps already expressed in the export calendar.
    notes['period'] = pd.to_datetime(notes.start_at, errors='raise').dt.to_period('M')
    end = pd.Period(as_of, freq='M')
    periods = list(pd.period_range(notes.period.min(), end, freq='M'))
    if len(periods) < 4:
        raise ValueError('notes: need at least four calendar months')
    counts = notes.groupby(['service_id', 'period']).size()
    first = notes.groupby('client_id').period.transform('min')
    first_counts = notes[notes.period == first].groupby(['service_id', 'period']).size()
    aggregates = notes.groupby(['service_id', 'period']).total_price.agg(['sum', 'mean'])
    material_price = materials.set_index('id').price.fillna(0).to_dict()
    costs = {}
    for row in norms.to_dict('records'):
        if row['material_id'] not in material_price:
            raise ValueError('service_materials: unknown material_id')
        if row['service_id'] not in set(services.id):
            raise ValueError('service_materials: unknown service_id')
        costs[row['service_id']] = costs.get(row['service_id'], 0) + row['required_quantity'] * material_price[row['material_id']]
    material_rows, price_rows = [], []
    for service in services.to_dict('records'):
        sid, price = service['id'], service['price']
        if not price > 0:
            raise ValueError(f'services: price must be positive for id={sid}')
        values = [int(counts.get((sid, p), 0)) for p in periods]
        average = sum(values) / len(values)
        ratios = [v / average for v in values] if average else []
        for i, period in enumerate(periods):
            angle = 2 * math.pi * period.month / 12
            if average and i >= 3:
                lags = [ratios[i-j] for j in (1, 2, 3)]
                prev = ratios[max(0, i-6):i-3]
                recent = sum(lags) / 3
                trend = recent / (sum(prev)/len(prev)) if prev and sum(prev) > 0 else 1.0
                material_rows.append([sid, values[i], ruby_round(average, 2), ruby_round(ratios[i], 4),
                    *[ruby_round(v, 4) for v in lags], ruby_round(recent, 4), ruby_round(trend, 4),
                    ruby_round(math.sin(angle), 4), ruby_round(math.cos(angle), 4), period.year, period.month])
            maximum = max(int(counts.get((s, period), 0)) for s in services.id) or 1
            visits = int(first_counts.get((sid, period), 0))
            share = visits / values[i] if values[i] else 0
            revenue, actual = (aggregates.loc[(sid, period), ['sum', 'mean']]
                               if (sid, period) in aggregates.index else (0, 0))
            cost = costs.get(sid, 0)
            target = min(price*PRICE_MAX_RATIO, max(price*PRICE_MIN_RATIO, price*(1+PRICE_DEMAND_WEIGHT*values[i]/maximum-PRICE_FIRST_VISIT_WEIGHT*share)))
            active = str(service['active']).lower() in ('true', '1', '1.0')
            price_rows.append([sid, price, int(active), values[i], ruby_round(revenue, 2), ruby_round(actual, 2),
                               ruby_round(share, 4), period.year, period.month, ruby_round(cost, 2), ruby_round(cost/price, 4), ruby_round(target, 2)])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(material_rows, columns='service_id total_bookings avg_bookings booking_ratio bookings_lag_1 bookings_lag_2 bookings_lag_3 avg_bookings_last_3m usage_trend month_sin month_cos year month'.split()).to_csv(output/'material_forecast.csv', index=False)
    # Ruby emits month-major price rows.
    pd.DataFrame(price_rows, columns='service_id base_price is_active total_bookings revenue avg_price_fact first_visit_share year month avg_material_cost_per_service material_cost_share target_price'.split()).sort_values(['year', 'month'], kind='stable').to_csv(output/'price_analysis.csv', index=False)
    exported = materials.rename(columns={'quantity': 'current_quantity'})[['id', 'title', 'current_quantity', 'minimal_quantity', 'price']].copy()
    exported['price'] = exported.price.fillna(0)
    (output/'materials.json').write_text(exported.to_json(orient='records', force_ascii=False), encoding='utf-8')
    (output/'service_materials.json').write_text(norms[contracts['service_materials']].to_json(orient='records'), encoding='utf-8')
    (output/'preparation.json').write_text(json.dumps({'mode': 'legacy-parity', 'as_of': str(as_of), 'includes_partial_month': True}), encoding='utf-8')


def prepare(data_dir, output_dir, as_of, *, mode='demand-v2'):
    """Prepare full-month demand by default; legacy parity is opt-in."""
    if mode == 'legacy-parity':
        return prepare_legacy(data_dir, output_dir, as_of)
    if mode != 'demand-v2':
        raise ValueError('mode: expected demand-v2 or legacy-parity')
    from .demand import prepare_monthly
    tables = load_raw(data_dir, fractional_quantities=True)
    monthly, metadata = prepare_monthly(tables, as_of)
    output = Path(output_dir)
    # Reuse price recommendation preparation, bounded by the same full months.
    # It retains its formula; its labels do not claim future-price forecasting.
    prepare_legacy(data_dir, output, metadata['last_complete_month']+'-01', tables=tables)
    (output/'material_forecast.csv').unlink(missing_ok=True)
    monthly.to_csv(output/'monthly_demand.csv', index=False)
    materials = tables['materials'].rename(columns={'quantity': 'current_quantity'}).copy()
    materials['price'] = materials.price.fillna(0)
    (output/'materials.json').write_text(materials[
        ['id', 'title', 'current_quantity', 'minimal_quantity', 'price', 'unit', 'purchase_step']
    ].to_json(orient='records', force_ascii=False), encoding='utf-8')
    (output/'preparation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
