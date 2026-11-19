"""Original formula-derived price recommendation scenario."""
import numpy as np
from ..config import (PRICE, PRICE_MIN_RATIO, PRICE_MAX_RATIO, PRICE_DEMAND_WEIGHT,
                      PRICE_FIRST_VISIT_WEIGHT, PRICE_UNPOPULAR_THRESHOLD, PRICE_UNPOPULAR_DISCOUNT)


def prepare_features(df):
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['price_ratio'] = df['target_price'] / df['base_price']
    return df


def predict(df, predictor):
    if predictor.features != PRICE.features:
        raise ValueError('price checkpoint: incompatible feature schema')
    features = list(predictor.features)
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Use only the latest year for prediction
    max_year = int(df['year'].max())
    df_predict = df[df['year'] == max_year]
    df_train = df[df['year'] < max_year]

    # Median bookings from training period for "unpopular" threshold
    median_bookings = df_train.groupby('service_id')['total_bookings'].mean().median()

    results = []

    for service_id in sorted(df_predict['service_id'].unique()):
        sdf = df_predict[df_predict['service_id'] == service_id]

        base_price = float(sdf['base_price'].iloc[-1])
        is_active = int(sdf['is_active'].iloc[-1])

        # Average latest-year features as model input
        X = sdf[features].mean().values.astype(np.float32)
        price_ratio = predictor.predict(X.reshape(1, -1)).item()

        suggested = base_price * price_ratio

        # Clamp +-15%
        suggested = max(base_price * PRICE_MIN_RATIO, min(base_price * PRICE_MAX_RATIO, suggested))

        # -5% when latest-year average bookings are below half the historical median
        avg_bookings = sdf['total_bookings'].mean()
        if avg_bookings < median_bookings * PRICE_UNPOPULAR_THRESHOLD:
            suggested *= PRICE_UNPOPULAR_DISCOUNT
            suggested = max(base_price * PRICE_MIN_RATIO, suggested)

        results.append({
            'service_id': int(service_id),
            'current_price': base_price,
            'suggested_price': round(suggested, 2),
            'difference': round(suggested - base_price, 2),
            'difference_pct': round((suggested - base_price) / base_price * 100, 1),
            'is_active': is_active,
        })

    return results


def split_data(df):
    df = prepare_features(df)
    df = df[df.year < df.year.max()]
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    df = df.assign(ym=df.year * 12 + df.month)
    months = sorted(df.ym.unique())
    if len(months) < 7:
        raise ValueError('price: need at least 7 training months')
    test_start, val_start = months[-3], months[-6]
    parts = (df[df.ym < val_start], df[(df.ym >= val_start) & (df.ym < test_start)],
             df[df.ym >= test_start])
    if any(part.empty for part in parts):
        raise ValueError('training: empty train/validation/test split')
    return parts


def price_values(part, ratios):
    base = part.base_price.to_numpy(dtype=float)
    return np.clip(np.asarray(ratios).reshape(-1), PRICE_MIN_RATIO, PRICE_MAX_RATIO)*base


def formula_values(part):
    maximum = part.groupby(['year', 'month']).total_bookings.transform('max').replace(0, 1)
    ratio = 1+PRICE_DEMAND_WEIGHT*part.total_bookings/maximum-PRICE_FIRST_VISIT_WEIGHT*part.first_visit_share
    return np.round(price_values(part, ratio), 2)


def control_predictions(frame, payloads, manifests, *, history=None):
    """Validate and forecast labelled control observations using saved models."""
    from ..models import Predictor
    from ..evaluation import period_provenance, assert_control_independent
    from ..config import PRICE
    from .serviceflow import PRICE_COLUMNS, number
    if set(PRICE_COLUMNS)-set(frame):
        raise ValueError('control price table must contain all price_analysis columns')
    for name in PRICE_COLUMNS:
        number(frame, 'control', name, integer=name in ('service_id','year','month','total_bookings','is_active'),
               positive=name in ('service_id','year','month','base_price'))
    if (frame.month > 12).any() or (frame.year > 9999).any() or frame.duplicated(['service_id','year','month']).any() or not frame.is_active.isin([0,1]).all() or (frame.first_visit_share > 1).any():
        raise ValueError('control price: invalid calendar, flags, share or duplicate service/month')
    provenance = period_provenance(frame)
    assert_control_independent(provenance, manifests)
    frame = prepare_features(frame)
    actual = frame.target_price.to_numpy()
    predicted = [price_values(frame, Predictor(state).predict(frame[list(PRICE.features)].values)) for state in payloads]
    identities = frame[['service_id', 'year', 'month']].astype(int).to_dict('records')
    conditions = dict(policy='monthly formula-recommendation reconstruction, clipping 85–115%', units='currency')
    return frame, actual, predicted, identities, conditions, provenance, None
