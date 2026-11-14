"""Original formula-derived price recommendation scenario."""
import numpy as np
from ..config import PRICE


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
        suggested = max(base_price * 0.85, min(base_price * 1.15, suggested))

        # -5% when latest-year average bookings are below half the historical median
        avg_bookings = sdf['total_bookings'].mean()
        if avg_bookings < median_bookings * 0.5:
            suggested *= 0.95
            suggested = max(base_price * 0.85, suggested)

        results.append({
            'service_id': int(service_id),
            'current_price': base_price,
            'suggested_price': round(suggested, 2),
            'difference': round(suggested - base_price, 2),
            'difference_pct': round((suggested - base_price) / base_price * 100, 1),
            'is_active': is_active,
        })

    return results
