"""Causal monthly demand, fixed-origin backtesting and persisted history."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ScenarioConfig
from .contracts import read_csv, number
from .metrics import regression_metrics
from .purchasing import validate_catalog

FEATURES = ('demand_lag_1', 'demand_lag_2', 'demand_lag_3',
            'demand_mean_3', 'demand_trend', 'month_sin', 'month_cos')
CONFIG = ScenarioConfig(FEATURES, 'total_bookings', (32, 16), 1500)
COLUMNS = ['service_id', 'year', 'month', 'total_bookings']


def prepare_monthly(tables, as_of):
    try:
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff) or cutoff.tzinfo is not None:
            raise ValueError('invalid cutoff')
    except (ValueError, TypeError) as exc:
        raise ValueError('as_of: expected timezone-free date') from exc
    # as_of is an exclusive information boundary; its month is unfinished.
    last = cutoff.to_period('M')-1
    notes = tables['notes']
    notes = notes[(notes.status == 2) & (notes.start_at < cutoff.to_period('M').start_time)].copy()
    if notes.empty:
        raise ValueError('notes: no completed visits in full months before as_of')
    notes['period'] = notes.start_at.dt.to_period('M')
    periods = pd.period_range(notes.period.min(), last, freq='M')
    if len(periods) < 4:
        raise ValueError('monthly_demand: need at least four complete months')
    materials = tables['materials']
    for field in ('unit', 'purchase_step'):
        if field not in materials:
            raise ValueError(f'materials.{field}: required for demand-v2')
    exported = materials.rename(columns={'quantity': 'current_quantity'}).copy()
    exported['price'] = exported.price.fillna(0)
    validate_catalog(exported.to_dict('records'), tables['service_materials'].to_dict('records'), set(tables['services'].id))
    counts = notes.groupby(['service_id', 'period']).size()
    rows = [dict(service_id=int(sid), year=int(p.year), month=int(p.month),
                 total_bookings=int(counts.get((sid, p), 0)))
            for sid in sorted(tables['services'].id) for p in periods]
    metadata = dict(mode='demand-v2', contract_version=2, as_of=cutoff.isoformat(), frequency='MS',
                    group_column='service_id', target='total_bookings',
                    last_complete_month=str(last), first_month=str(periods[0]), includes_partial_month=False,
                    missing_month_policy='zero', horizon_max=3)
    return pd.DataFrame(rows, columns=COLUMNS), metadata


def validate_monthly(frame):
    frame = frame.copy()
    if frame.empty or set(COLUMNS)-set(frame):
        raise ValueError('monthly_demand: required nonempty service_id/year/month/total_bookings')
    for column in COLUMNS:
        number(frame, 'monthly_demand', column, integer=True, positive=column != 'total_bookings')
    if (frame.month > 12).any() or (frame.year > 9998).any():
        raise ValueError('monthly_demand: invalid year/month')
    if frame.duplicated(['service_id', 'year', 'month']).any():
        raise ValueError('monthly_demand: duplicate service/month')
    frame['ym'] = frame.year*12+frame.month
    start, end = int(frame.ym.min()), int(frame.ym.max())
    expected = set(range(start, end+1))
    for sid, part in frame.groupby('service_id'):
        if set(part.ym) != expected:
            raise ValueError(f'monthly_demand: service {sid} must have a complete common monthly calendar; fill confirmed zero months explicitly')
    if end-start+1 < 4:
        raise ValueError('monthly_demand: need at least four complete months')
    return frame.sort_values(['service_id', 'ym']).reset_index(drop=True)


def load_monthly(directory):
    directory = Path(directory)
    try:
        metadata = json.loads((directory/'preparation.json').read_text(encoding='utf-8'))
        cutoff = pd.Timestamp(metadata['as_of'])
        if (metadata['mode'] != 'demand-v2' or metadata['frequency'] != 'MS'
                or metadata.get('contract_version') != 2 or metadata.get('target') != 'total_bookings'
                or metadata.get('group_column') != 'service_id' or metadata.get('includes_partial_month') is not False
                or pd.isna(cutoff) or cutoff.tzinfo is not None):
            raise ValueError('invalid metadata')
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('preparation.json: expected demand-v2 monthly metadata with as_of') from exc
    frame = validate_monthly(read_csv(directory/'monthly_demand.csv', COLUMNS))
    last = pd.Period(year=int(frame.year.max()), month=int(frame[frame.year == frame.year.max()].month.max()), freq='M')
    if last != cutoff.to_period('M')-1 or str(last) != metadata.get('last_complete_month'):
        raise ValueError('monthly_demand: last month must match complete-month cutoff')
    return frame, metadata


def feature_vector(history, month):
    """Same rule for training, validation and recursive forecast; history only."""
    if len(history) < 3:
        raise ValueError('demand features: need three preceding months')
    lag1, lag2, lag3 = history[-1], history[-2], history[-3]
    recent = (lag1+lag2+lag3)/3
    previous = history[max(0, len(history)-6):len(history)-3]
    previous_mean = sum(previous)/len(previous) if previous else 0
    trend = recent/previous_mean if previous_mean > 0 else 1.0
    angle = 2*np.pi*month/12
    return [lag1, lag2, lag3, recent, trend, np.sin(angle), np.cos(angle)]


def feature_table(frame):
    frame = validate_monthly(frame)
    rows = []
    for sid, part in frame.groupby('service_id', sort=True):
        history = []
        for row in part.to_dict('records'):
            if len(history) >= 3:
                rows.append({**row, **dict(zip(FEATURES, feature_vector(history, row['month'])))})
            history.append(row['total_bookings'])
    return pd.DataFrame(rows)


def nonnegative(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('demand model: non-finite prediction')
    return max(0.0, value)


def forecast(history_frame, predictor, months=3, *, baseline=None):
    if type(months) is not int or months not in (1, 2, 3):
        raise ValueError('months must be 1, 2 or 3')
    frame = validate_monthly(history_frame)
    if baseline not in (None, 'last_value', 'mean3'):
        raise ValueError('unknown demand baseline')
    if baseline is None and predictor.features != FEATURES:
        raise ValueError('demand-v2 checkpoint: incompatible feature schema')
    origin = int(frame.ym.max())
    result = []
    for sid, part in frame.groupby('service_id', sort=True):
        history = part.total_bookings.astype(float).tolist()
        for lead in range(1, months+1):
            year, month = divmod(origin+lead-1, 12)
            month += 1
            features = feature_vector(history, month)
            if baseline == 'last_value':
                value = history[-1]
            elif baseline == 'mean3':
                value = sum(history[-3:])/3
            else:
                value = predictor.predict(np.array([features], dtype=np.float32)).item()
            value = nonnegative(value)
            history.append(value)
            result.append(dict(service_id=int(sid), year=year, month=month, lead=lead,
                               predicted_bookings=value))
    return result


def evaluate(frame, predictor, start, *, months=3):
    """One origin before start. Actual values are used ONLY for scoring."""
    frame = validate_monthly(frame)
    history = frame[frame.ym < start]
    actuals = {(int(r.service_id), int(r.year), int(r.month)): float(r.total_bookings)
               for r in frame[frame.ym >= start].itertuples()}
    report = dict(origin=month_label(start-1), start=month_label(start), horizon=months,
                  units='completed_bookings', policy='fixed-origin-recursive', models={})
    for label, baseline in [('mlp', None), ('last_value', 'last_value'), ('mean3', 'mean3')]:
        predictions = forecast(history, predictor, months, baseline=baseline)
        rows = []
        for row in predictions:
            key = (row['service_id'], row['year'], row['month'])
            if key not in actuals:
                raise ValueError('evaluation: insufficient observed months for requested horizon')
            rows.append({**row, 'actual': actuals[key]})
        metrics = regression_metrics([r['actual'] for r in rows], [r['predicted_bookings'] for r in rows])
        by_lead = {str(lead): regression_metrics([r['actual'] for r in rows if r['lead'] == lead],
                                                [r['predicted_bookings'] for r in rows if r['lead'] == lead])
                   for lead in range(1, months+1)}
        report['models'][label] = dict(metrics=metrics, by_lead=by_lead, rows=rows)
    return report



def evaluate_one_step(frame, predictor, start):
    table = feature_table(frame)
    table = table[table.ym >= start]
    predicted = np.maximum(predictor.predict(table[list(FEATURES)].values).reshape(-1), 0)
    report = dict(policy='rolling-one-step-observed-history', units='completed_bookings', models={})
    for name, values in [('mlp', predicted), ('last_value', table.demand_lag_1.values),
                         ('mean3', table.demand_mean_3.values)]:
        rows = [dict(service_id=int(row.service_id), year=int(row.year), month=int(row.month),
                     actual=float(row.total_bookings), predicted_bookings=float(value))
                for row, value in zip(table.itertuples(), values)]
        report['models'][name] = dict(metrics=regression_metrics(table.total_bookings.values, values), rows=rows)
    return report


def month_label(ym):
    year, month = divmod(int(ym)-1, 12)
    return f'{year:04d}-{month+1:02d}'


def fit_demand(frame, metadata, settings):
    from .training import fit, split_data
    from .inference import Predictor
    frame = validate_monthly(frame)
    try:
        cutoff = pd.Timestamp(metadata['as_of'])
        if pd.isna(cutoff) or cutoff.tzinfo is not None:
            raise ValueError('invalid cutoff')
        expected = cutoff.to_period('M')-1
        if month_label(frame.ym.max()) != str(expected):
            raise ValueError('incomplete or missing final month')
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('demand training: history must end at the complete-month as_of boundary') from exc
    features = feature_table(frame)
    train, validation, test = split_data(features, 'materials')
    val_start, test_start = int(validation.ym.min()), int(test.ym.min())
    def recursive_validation(state):
        report = evaluate(frame[frame.ym < test_start], Predictor(state), val_start)
        return report['models']['mlp']['metrics']['mse']
    state = fit(features, 'materials', CONFIG, settings,
                validation_evaluator=recursive_validation, prediction_transform=lambda values: np.maximum(values, 0))
    predictor = Predictor(state)
    # One-step metrics use actual previous observations; label them explicitly.
    best_val_loss = state['metrics']['best_val_loss']
    one_step = evaluate_one_step(frame, predictor, test_start)
    report = evaluate(frame, predictor, test_start)
    bounds = {name: dict(rows=len(part), start=month_label(part.ym.min()), end=month_label(part.ym.max()))
              for name, part in [('train', train), ('validation', validation), ('test', test)]}
    state['metrics'] = dict(one_step=one_step, recursive_test=report,
                            best_val_loss=best_val_loss, validation_policy='fixed-origin-recursive-mse',
                            recursive_validation=evaluate(frame[frame.ym < test_start], predictor, val_start),
                            split=bounds, units='completed_bookings')
    state['demand_context'] = dict(contract_version=2, frequency='MS', target='total_bookings',
                                   feature_policy='causal-count-lags-v2', horizon_max=3,
                                   as_of=metadata['as_of'], split=bounds,
                                   history=frame[COLUMNS].astype(int).to_dict('records'))
    return state
