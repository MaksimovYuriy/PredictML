"""Python translation of the two Ruby export services (legacy parity)."""
from pathlib import Path
import json
import math
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd


def ruby_round(value, digits):
    """Decimal half-away-from-zero for the Ruby export boundary.

    Internal calculations remain floats; this does not promise bitwise parity
    for every possible binary floating-point boundary.
    """
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def prepare_legacy(data_dir, output_dir, as_of, *, tables=None):
    data, output = Path(data_dir), Path(output_dir)
    from .contracts import load_raw, RAW_COLUMNS
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
            target = min(price*1.15, max(price*.85, price*(1+.05*values[i]/maximum-.03*share)))
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
    from .contracts import load_raw
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
