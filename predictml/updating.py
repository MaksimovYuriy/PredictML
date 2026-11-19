"""Scratch retraining and continuation with a frozen input space."""
from dataclasses import asdict, replace
import copy
import numpy as np
import torch
from .config import PRICE, TrainingConfig, DEFAULT_SEED, DEFAULT_HIDDEN_SIZES
from .training import fit_arrays, fit_estimator
from .models import Predictor
from .evaluation import regression_metrics, tabular_provenance, period_provenance


def optimize(arrays, targets, parent, settings, operation, features, mean, std, hidden_sizes=None, objective=None):
    kind = parent.get('model_kind', 'mlp')
    layers = tuple(hidden_sizes or parent.get('hidden_sizes', DEFAULT_HIDDEN_SIZES))
    if operation == 'retrain':
        return fit_estimator(arrays[0], targets[0], arrays[1], targets[1], arrays[2], kind,
                             layers, settings, features, mean, std, validation_evaluator=objective)
    if kind != 'mlp':
        raise ValueError('fine-tune is supported only for MLP; use retrain for linear/baseline')
    if hidden_sizes is not None and tuple(hidden_sizes) != tuple(parent['hidden_sizes']):
        raise ValueError('fine-tune cannot change architecture')
    settings = replace(settings, dropout=parent.get('dropout', TrainingConfig.dropout))
    log = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings.seed if settings.seed is not None else DEFAULT_SEED)
        weights, loss, predicted = fit_arrays(
            arrays[0], targets[0], arrays[1], targets[1], arrays[2], parent['hidden_sizes'], settings,
            validation_evaluator=objective, features=list(features), mean=np.asarray(mean), std=np.asarray(std),
            training_log=log, initial_state=parent['model_state'])
    state = copy.deepcopy(parent)
    state['model_state'] = weights
    # The previous optimizer was not stored by v1; a fresh Adam is explicit.
    log.update(validation_mse=loss, settings=asdict(settings), optimizer_policy='fresh Adam; continue saved weights',
               preprocessing_policy='frozen', initialized_from_parent=True)
    return state, predicted, log


def tabular_update(parent, frame, settings, operation, config, hidden_sizes):
    from .training import train_table, split_rows
    from .prediction import TabularModel
    from .data import TableConfig, numeric
    if operation == 'retrain':
        cfg = config or dict(target=parent.schema['target'], features=parent.schema['features'],
                             types=parent.schema['types'], temporal=parent.schema['temporal'],
                             exclude=[c for c in parent.schema['excluded'] if c in frame], seed=DEFAULT_SEED if settings.seed is None else settings.seed)
        model, _ = train_table(frame, cfg, settings, tuple(hidden_sizes or parent.checkpoint.get('hidden_sizes', DEFAULT_HIDDEN_SIZES)),
                               parent.checkpoint.get('model_kind', 'mlp'))
        kind = model.report['selected_model']
        model.checkpoints = {kind: model.checkpoint}
        return model
    if config is not None:
        raise ValueError('fine-tune uses the saved schema; use retrain to change target/features')
    cfg = TableConfig(target=parent.schema['target'], features=tuple(parent.schema['features']),
                      types=parent.schema['types'], temporal=parent.schema['temporal'], seed=DEFAULT_SEED if settings.seed is None else settings.seed)
    if cfg.target not in frame:
        raise ValueError(f'target: missing column {cfg.target}')
    target = numeric(frame[cfg.target], cfg.target)
    keep = target.notna()
    working, y = frame.loc[keep].reset_index(drop=True), target.loc[keep].reset_index(drop=True)
    parts, split = split_rows(working, cfg)
    preprocessor = copy.deepcopy(parent.preprocessor)
    arrays = [preprocessor.transform(working.iloc[part]) for part in parts]
    targets = [y.iloc[part].to_numpy(dtype=np.float32).reshape(-1, 1) for part in parts]
    if any(not np.isfinite(a).all() for a in targets):
        raise ValueError('target exceeds float32 range')
    state, predicted, log = optimize(arrays, targets, parent.checkpoint, settings, operation,
                                      preprocessor.output_names, parent.checkpoint['feature_mean'],
                                      parent.checkpoint['feature_std'], hidden_sizes)
    report = dict(settings=log['settings'], seed=cfg.seed, input_rows=len(frame), dropped_missing_target=int((~keep).sum()),
                  split={**split, 'parts': {name: dict(rows=len(part), source_rows=np.flatnonzero(keep)[part].tolist())
                                           for name, part in zip(('train', 'validation', 'test'), parts)}},
                  selected_model='mlp', models={'mlp': {**log, 'test': regression_metrics(targets[2], predicted)}},
                  test=regression_metrics(targets[2], predicted), training=log)
    model = TabularModel(copy.deepcopy(parent.schema), preprocessor, state, report)
    model.checkpoints = {'mlp': state}
    model.provenance = tabular_provenance(working, model.schema)
    return model


def checkpoint_update(parent, task, data, settings, operation, hidden_sizes):
    from .scenarios.demand import (load_monthly, feature_table, CONFIG, evaluate, evaluate_one_step, month_label, COLUMNS)
    if task == 'materials':
        if not parent.get('demand_context'):
            raise ValueError('updates require demand-v2; retrain legacy data as a new demand-v2 model')
        frame, metadata = load_monthly(data)
        if set(frame.service_id) != {r['service_id'] for r in parent['demand_context']['history']}:
            raise ValueError('service set changed; train a new model')
        table, config = feature_table(frame), CONFIG
    else:
        from .scenarios.serviceflow import validate_prepared
        frame, metadata, config = validate_prepared(data, 'price', True), {}, PRICE
        table = frame
    if task == 'materials':
        from .scenarios.demand import split_data
    else:
        from .scenarios.price import split_data
    parts = split_data(table)
    raw = [part[list(config.features)].values.astype(np.float32) for part in parts]
    targets = [part[config.target].values.astype(np.float32).reshape(-1, 1) for part in parts]
    mean = raw[0].mean(axis=0) if operation == 'retrain' else np.asarray(parent['feature_mean'], dtype=np.float32)
    std = raw[0].std(axis=0) if operation == 'retrain' else np.asarray(parent['feature_std'], dtype=np.float32)
    std[std == 0] = 1
    arrays = [(a-mean)/std for a in raw]
    if task == 'materials':
        val_start, test_start = int(parts[1].ym.min()), int(parts[2].ym.min())
        def objective(state):
            report = evaluate(frame[frame.ym < test_start], Predictor(state), val_start)
            return report['models']['mlp']['metrics']['mse']
    else:
        from .scenarios.price import price_values
        def objective(state):
            predicted = price_values(parts[1], Predictor(state).predict(raw[1]))
            return regression_metrics(parts[1].target_price, predicted)['mse']
    state, predicted, log = optimize(arrays, targets, parent, settings, operation, config.features, mean, std,
                                      hidden_sizes, objective)
    if task == 'materials':
        kind = state.get('model_kind', 'mlp')
        bounds = {name: dict(rows=len(part), start=month_label(part.ym.min()), end=month_label(part.ym.max()))
                  for name, part in zip(('train', 'validation', 'test'), parts)}
        state['demand_context'] = {**parent['demand_context'], 'as_of': metadata['as_of'], 'split': bounds,
                                   'history': frame[COLUMNS].astype(int).to_dict('records')}
        predictor = Predictor(state)
        state['metrics'] = dict(training=log, split=bounds,
                                recursive_validation=evaluate(frame[frame.ym < test_start], predictor, val_start, model_name=kind),
                                recursive_test=evaluate(frame, predictor, test_start, model_name=kind),
                                one_step=evaluate_one_step(frame, predictor, test_start, model_name=kind), units='completed_bookings')
    else:
        state['price_context'] = parent.get('price_context') or dict(task='formula-derived-price-recommendation', units='currency')
        state['metrics'] = dict(training=log, validation=regression_metrics(parts[1].target_price, price_values(parts[1], Predictor(state).predict(raw[1]))),
                                test=regression_metrics(parts[2].target_price, price_values(parts[2], Predictor(state).predict(raw[2]))), units='currency')
    return state, period_provenance(frame)
