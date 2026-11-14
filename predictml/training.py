"""Shared trainer; preserves the original optimization and split policies."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .models import MLP
from .metrics import compute_metrics
from .scenarios.price import prepare_features


def split_data(df, scenario):
    if scenario == 'materials':
        df = df.sort_values(['service_id', 'year', 'month']).reset_index(drop=True)
        df = df.assign(ym=df.year * 12 + df.month)
        test_start = int(df.ym.max()) - 2
        val_start = test_start - 3
    elif scenario == 'price':
        df = prepare_features(df)
        df = df[df.year < df.year.max()]
        df = df.sort_values(['year', 'month']).reset_index(drop=True)
        df = df.assign(ym=df.year * 12 + df.month)
        months = sorted(df.ym.unique())
        if len(months) < 7:
            raise ValueError('price: need at least 7 training months')
        test_start, val_start = months[-3], months[-6]
    else:
        raise ValueError('unknown scenario')
    parts = (df[df.ym < val_start], df[(df.ym >= val_start) & (df.ym < test_start)],
             df[df.ym >= test_start])
    if any(part.empty for part in parts):
        raise ValueError('training: empty train/validation/test split')
    return parts


def baseline_metrics(test_df, targets, scenario, mae):
    if scenario == 'materials':
        _, lag_mae, lag_wmape = compute_metrics(targets, test_df.bookings_lag_1.values.astype(np.float32))
        _, avg_mae, avg_wmape = compute_metrics(targets, test_df.avg_bookings_last_3m.values.astype(np.float32))
        best = min(lag_mae, avg_mae)
        result = dict(baseline_lag1_mae=lag_mae, baseline_lag1_wmape=lag_wmape,
                      baseline_avg3_mae=avg_mae, baseline_avg3_wmape=avg_wmape)
    else:
        mse, best, wmape = compute_metrics(targets, np.ones_like(targets))
        result = dict(baseline_mse=mse, baseline_mae=best, baseline_wmape=wmape)
    result['mae_improvement_pct'] = (best-mae)/best*100 if best > 0 else 0.0
    return result


def fit(df, scenario, model_config, training_config):
    """Return a legacy-compatible checkpoint; no IO or global mutations."""
    train, validation, test = split_data(df, scenario)
    features = list(model_config.features)
    def arrays(part):
        return (part[features].values.astype(np.float32),
                part[model_config.target].values.astype(np.float32).reshape(-1, 1))
    x_train, y_train = arrays(train)
    x_val, y_val = arrays(validation)
    x_test, y_test = arrays(test)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1
    x_train, x_val, x_test = [(x-mean)/std for x in (x_train, x_val, x_test)]
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
                        batch_size=training_config.batch_size, shuffle=True)
    model = MLP(x_train.shape[1], model_config.hidden_sizes, training_config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.MSELoss()
    best_loss, best_state = float('inf'), None
    for epoch in range(training_config.epochs):
        model.train()
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % training_config.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(torch.tensor(x_val)), torch.tensor(y_val)).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise ValueError('training: no finite validation loss; check input data')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(x_test)).numpy()
    mse, mae, wmape = compute_metrics(y_test, predictions)
    metrics = dict(best_val_loss=best_loss, test_mse=mse, test_mae=mae, test_wmape=wmape,
                   train_rows=len(train), val_rows=len(validation), test_rows=len(test))
    metrics.update(baseline_metrics(test, y_test, scenario, mae))
    return dict(model_state=best_state, input_size=x_train.shape[1],
                hidden_sizes=list(model_config.hidden_sizes), dropout=training_config.dropout,
                feature_mean=mean.tolist(), feature_std=std.tolist(), features=features, metrics=metrics)
