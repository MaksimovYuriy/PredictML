"""Split and train models; this module does not read files or select scenarios."""
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
import uuid
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from .config import DEFAULT_SEED, DEFAULT_HIDDEN_SIZES, TABLE_TRAINING_DEFAULTS
from .data import TableConfig, numeric, dates
from .preprocessing import Preprocessor
from .models import MLP, Predictor
from .prediction import TabularModel
from .evaluation import compute_metrics, regression_metrics, tabular_provenance


def fit_arrays(x_train, y_train, x_val, y_val, x_test, hidden_sizes, training_config,
               *, validation_evaluator=None, mean=None, std=None, features=None, training_log=None, initial_state=None):
    """Shared MLP optimization for already prepared float32 arrays."""
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
                        batch_size=training_config.batch_size, shuffle=True)
    model = MLP(x_train.shape[1], hidden_sizes, training_config.dropout)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.MSELoss()
    best_loss, best_state = float('inf'), None
    patience_loss, stale, best_epoch = float('inf'), 0, None
    curve = []
    for epoch in range(training_config.epochs):
        model.train()
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % training_config.validation_interval == 0:
            model.eval()
            if validation_evaluator is None:
                with torch.no_grad():
                    val_loss = criterion(model(torch.tensor(x_val)), torch.tensor(y_val)).item()
            else:
                candidate = dict(model_state=model.state_dict(), input_size=x_train.shape[1],
                                 hidden_sizes=list(hidden_sizes), dropout=training_config.dropout,
                                 feature_mean=mean.tolist(), feature_std=std.tolist(), features=features)
                val_loss = validation_evaluator(candidate)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
                best_epoch = epoch+1
            if val_loss < patience_loss-training_config.min_delta:
                patience_loss, stale = val_loss, 0
            else:
                stale += 1
            curve.append(dict(epoch=epoch+1, validation_mse=float(val_loss)))
            if training_config.patience is not None and stale >= training_config.patience:
                break
    if best_state is None:
        raise ValueError('training: no finite validation loss; check input data')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(x_test)).numpy()
    if training_log is not None:
        training_log.update(best_epoch=best_epoch, epochs_ran=epoch+1,
                            stopped_early=epoch+1 < training_config.epochs, curve=curve)
    return best_state, best_loss, predictions


def fit_estimator(x_train, y_train, x_val, y_val, x_test, kind, hidden_sizes,
                  settings, features, mean, std, *, validation_evaluator=None, seed=None):
    started = perf_counter()
    log = {}
    state = dict(input_size=x_train.shape[1], features=list(features),
                 feature_mean=np.asarray(mean).tolist(), feature_std=np.asarray(std).tolist(), model_kind=kind)
    if kind == 'mlp':
        if not hidden_sizes or any(type(n) is not int or n < 1 for n in hidden_sizes):
            raise ValueError('hidden_sizes must contain positive integers')
        selected_seed = settings.seed if settings.seed is not None else seed
        if selected_seed is None:
            weights, best_loss, predicted = fit_arrays(
                x_train, y_train, x_val, y_val, x_test, hidden_sizes, settings,
                validation_evaluator=validation_evaluator, mean=mean, std=std, features=list(features), training_log=log)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(selected_seed)
                weights, best_loss, predicted = fit_arrays(
                    x_train, y_train, x_val, y_val, x_test, hidden_sizes, settings,
                    validation_evaluator=validation_evaluator, mean=mean, std=std, features=list(features), training_log=log)
        state.update(model_state=weights, hidden_sizes=list(hidden_sizes), dropout=settings.dropout)
    elif kind in ('linear', 'baseline'):
        if kind == 'linear':
            model = LinearRegression().fit(x_train.astype(float), y_train.reshape(-1).astype(float))
            coefficient, intercept = model.coef_, float(model.intercept_)
        else:
            coefficient, intercept = np.zeros(x_train.shape[1]), float(y_train.mean())
        state.update(coefficients=coefficient.tolist(), intercept=intercept)
        predicted = (x_test @ coefficient+intercept).reshape(-1, 1)
        val = (x_val @ coefficient+intercept).reshape(-1, 1)
        best_loss = validation_evaluator(state) if validation_evaluator else float(np.mean((val-y_val)**2))
    else:
        raise ValueError('model_kind must be mlp, linear or baseline')
    log.update(validation_mse=float(best_loss), fit_seconds=perf_counter()-started,
               settings=asdict(settings), hidden_sizes=list(hidden_sizes) if kind == 'mlp' else None)
    return state, predicted, log


def fit_candidates(arrays, targets, features, settings, hidden_sizes, *, seed=DEFAULT_SEED, validation_evaluator=None):
    states, predictions, details = {}, {}, {}
    for kind in ('baseline', 'linear', 'mlp'):
        state, predicted, log = fit_estimator(*[arrays[0], targets[0], arrays[1], targets[1], arrays[2]],
                                            kind, hidden_sizes, settings, features,
                                            np.zeros(arrays[0].shape[1]), np.ones(arrays[0].shape[1]), seed=seed,
                                            validation_evaluator=validation_evaluator)
        with torch.random.fork_rng(devices=[]):
            predictor = Predictor(state)
            validation = predictor.predict(arrays[1]).reshape(-1)
        details[kind] = {**log, 'validation': regression_metrics(targets[1].reshape(-1), validation),
                         'test': regression_metrics(targets[2].reshape(-1), predicted.reshape(-1))}
        states[kind], predictions[kind] = state, predicted.reshape(-1)
    # The test is never used to choose the recommended model.
    selected = min(details, key=lambda name: details[name]['validation']['mse'])
    return states, predictions, details, selected


def fit_prepared(parts, model_config, training_config, *, baseline_evaluator=None, validation_evaluator=None, prediction_transform=None, model_kind="mlp", training_log=None):
    """Return weights and train-only scaling; optional recursive validation."""
    train, validation, test = parts
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
    if model_kind != 'mlp' or training_config.seed is not None or training_log is not None:
        state, predictions, details = fit_estimator(
            x_train, y_train, x_val, y_val, x_test, model_kind, model_config.hidden_sizes,
            training_config, features, mean, std, validation_evaluator=validation_evaluator)
        best_loss = details['validation_mse']
        if training_log is not None:
            training_log.update(details)
    else:
        best_state, best_loss, predictions = fit_arrays(
            x_train, y_train, x_val, y_val, x_test, model_config.hidden_sizes, training_config,
            validation_evaluator=validation_evaluator, mean=mean, std=std, features=features)
        state = dict(model_state=best_state, input_size=x_train.shape[1],
                     hidden_sizes=list(model_config.hidden_sizes), dropout=training_config.dropout,
                     feature_mean=mean.tolist(), feature_std=std.tolist(), features=features)
    if prediction_transform is not None:
        predictions = prediction_transform(predictions)
    mse, mae, wmape = compute_metrics(y_test, predictions)
    metrics = dict(best_val_loss=best_loss, test_mse=mse, test_mae=mae, test_wmape=wmape,
                   train_rows=len(train), val_rows=len(validation), test_rows=len(test))
    if baseline_evaluator is not None:
        metrics.update(baseline_evaluator(test, y_test, mae, model_config.features))
    state['metrics'] = metrics
    return state


def split_rows(frame, config):
    if len(frame) < 10:
        raise ValueError('training: need at least 10 rows with known numeric target')
    if config.temporal is None:
        positions = np.random.default_rng(config.seed).permutation(len(frame))
        a, b = int(len(frame)*.6), int(len(frame)*.8)
        return (positions[:a], positions[a:b], positions[b:]), {'policy': 'random', 'seed': config.seed}
    settings = config.temporal
    required = {'time_column', 'frequency', 'horizon', 'target_definition', 'available_features'}
    if not isinstance(settings, dict) or required-set(settings):
        raise ValueError('temporal: specify time_column/frequency/horizon/target_definition/available_features')
    if set(settings)-required:
        raise ValueError('temporal: unknown fields')
    column = settings['time_column']
    if column not in frame:
        raise ValueError(f'temporal.time_column: missing {column!r}')
    if type(settings['horizon']) is not int or settings['horizon'] < 1:
        raise ValueError('temporal.horizon: expected positive number of periods')
    if not isinstance(settings['target_definition'], str) or not settings['target_definition'].strip():
        raise ValueError('temporal.target_definition: required description of prepared target')
    try:
        offset = pd.tseries.frequencies.to_offset(settings['frequency'])
        if offset.n <= 0:
            raise ValueError('nonpositive offset')
    except (ValueError, TypeError) as exc:
        raise ValueError('temporal.frequency: expected positive pandas frequency, e.g. D or MS') from exc
    time = dates(frame[column], column)
    if time.isna().any() or not time.map(offset.is_on_offset).all():
        raise ValueError(f'{column}: missing or off-frequency origin dates')
    unique = sorted(time.unique())
    if len(unique) < 10:
        raise ValueError('temporal: need at least 10 distinct forecast origins')
    val_start, test_start = unique[int(len(unique)*.6)], unique[int(len(unique)*.8)]
    label_time = time.map(lambda t: t+settings['horizon']*offset)
    train = np.flatnonzero(((time < val_start) & (label_time < val_start)).to_numpy())
    val = np.flatnonzero(((time >= val_start) & (time < test_start) & (label_time < test_start)).to_numpy())
    test = np.flatnonzero((time >= test_start).to_numpy())
    if len(train) < 2 or not len(val) or not len(test):
        raise ValueError('temporal: horizon leaves insufficient train/validation/test rows; extend history')
    return (train, val, test), dict(policy='chronological-prepared-target',
                                  val_start=str(val_start), test_start=str(test_start),
                                  purged_rows=len(frame)-len(train)-len(val)-len(test), **settings)


def train_table(frame, config, settings=None, hidden_sizes=DEFAULT_HIDDEN_SIZES, model_kind="mlp"):
    if not isinstance(config, TableConfig):
        config = TableConfig.from_dict(config)
    if frame.columns.duplicated().any():
        raise ValueError('duplicate column names')
    if any(not isinstance(c, str) or not c.strip() for c in frame.columns):
        raise ValueError('column names must be nonempty strings')
    if config.target not in frame:
        raise ValueError(f'target: missing column {config.target!r}')
    # Validate configuration early; types themselves are inferred only on train.
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError('seed: expected integer in [0, 2**32)')
    target = numeric(frame[config.target], config.target)
    keep = target.notna().to_numpy()
    positions = np.flatnonzero(keep)
    working = frame.iloc[positions].reset_index(drop=True)
    target = target.iloc[positions].reset_index(drop=True)
    parts, split = split_rows(working, config)
    train, validation, test = parts
    features, kinds = config.resolve(working.iloc[train])
    if config.temporal is not None:
        available = config.temporal['available_features']
        if not isinstance(available, list) or any(not isinstance(v, str) for v in available) or len(available) != len(set(available)) or set(available) != set(features):
            raise ValueError('temporal.available_features: explicitly confirm every selected feature is available at forecast origin')
    preprocessor = Preprocessor(features, kinds).fit(working.iloc[train])
    arrays = [preprocessor.transform(working.iloc[part]) for part in parts]
    targets = [target.iloc[part].to_numpy(dtype=np.float32).reshape(-1, 1) for part in parts]
    if any(not np.isfinite(y).all() for y in targets):
        raise ValueError(f'{config.target}: values exceed float32 range')
    settings = settings or TABLE_TRAINING_DEFAULTS
    states, predictions_by_model, comparison, recommended = fit_candidates(
        arrays, targets, preprocessor.output_names, settings, hidden_sizes, seed=config.seed)
    if model_kind == 'best':
        model_kind = recommended
    if model_kind not in states:
        raise ValueError('model_kind must be mlp, linear, baseline or best')
    checkpoint = states[model_kind]
    predictions = predictions_by_model[model_kind]
    best_loss = comparison[model_kind]['validation_mse']
    schema = dict(target=config.target, features=list(features), types=kinds,
                  excluded=list(config.exclude), temporal=config.temporal,
                  unknown_categories='all-zero encoding', extra_columns='ignored',
                  column_order='saved names', target_units='original')
    split['parts'] = {name: dict(rows=len(part), source_rows=positions[part].tolist())
                      for name, part in zip(('train', 'validation', 'test'), parts)}
    report = dict(run_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(), input_rows=len(frame), dropped_missing_target=int((~keep).sum()), split=split,
                  best_validation_mse=best_loss, test=regression_metrics(targets[2].reshape(-1), predictions.reshape(-1)),
                  settings=asdict(settings), seed=config.seed, transformed_features=len(preprocessor.output_names),
                  models=comparison, recommended_model=recommended, selected_model=model_kind,
                  selection_policy='minimum validation MSE; test is final evaluation')
    table = pd.DataFrame(dict(source_row=positions[test], actual=targets[2].reshape(-1), prediction=predictions.reshape(-1)))
    for name, values in predictions_by_model.items():
        table[f'prediction_{name}'] = values
    model = TabularModel(schema, preprocessor, checkpoint, report)
    model.provenance = tabular_provenance(working, schema)
    model.checkpoints = states
    return model, table
