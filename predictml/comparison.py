"""Comparable ServiceFlow regressors; no test-based model selection."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from .config import PRICE, TrainingConfig
from .inference import Predictor
from .metrics import regression_metrics
from .reporting import export_comparison


def _compare_service(scenario, data_dir, output_dir, *, settings=None, hidden_sizes=None, months=3, run_id=None):
    settings = settings or TrainingConfig(epochs=200, validation_interval=1, patience=20, seed=42)
    if scenario not in ('materials', 'price'):
        raise ValueError('scenario must be materials or price')
    if type(months) is not int or months not in (1, 2, 3):
        raise ValueError('months must be 1, 2 or 3')
    data, output = Path(data_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if scenario == 'materials':
        report, rows = compare_demand(data, output, settings, hidden_sizes, months)
    else:
        report, rows = compare_price(data, output, settings, hidden_sizes)
    report.update(run_id=run_id or str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
                  settings=asdict(settings), selection_policy='minimum validation MSE; test never used for selection')
    report['recommended_model'] = min(report['models'], key=lambda name: report['models'][name]['validation']['mse'])
    (output/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    export_comparison(output, report['models'], rows, report['split'], {k: v for k, v in report.items() if k not in ('models', 'split')})
    return report


def compare_demand(data, output, settings, hidden_sizes, months):
    from .demand import load_monthly, fit_demand, forecast
    from .purchasing import load_catalog, calculate
    frame, metadata = load_monthly(data)
    materials, norms = load_catalog(data, set(frame.service_id))
    models, rows, split = {}, [], None
    for kind in ('mlp', 'linear'):
        log = {}
        state = fit_demand(frame, metadata, settings, model_kind=kind,
                           hidden_sizes=hidden_sizes, training_log=log)
        torch.save(state, output/f'material-{kind}.pth')
        if kind == 'mlp':
            torch.save(state, output/'material_model.pth')
        val, test = state['metrics']['recursive_validation'], state['metrics']['recursive_test']
        split = state['metrics']['split']
        for name in ([kind, 'last_value', 'mean3'] if kind == 'mlp' else [kind]):
            models[name] = dict(validation=val['models'][name]['metrics'], test=test['models'][name]['metrics'],
                                by_lead=test['models'][name]['by_lead'],
                                training=log if name == kind else {'fit_seconds': 0, 'policy': 'history-only'})
            rows.extend({**row, 'prediction': row['predicted_bookings'], 'model': name, 'origin': test['origin']}
                        for row in test['models'][name]['rows'])
        demand = forecast(frame, Predictor(state), months)
        (output/f'demand-{kind}-predictions.json').write_text(json.dumps(demand, indent=2), encoding='utf-8')
        (output/f'material-{kind}-predictions.json').write_text(json.dumps(calculate(demand, materials, norms), indent=2, ensure_ascii=False), encoding='utf-8')
    return dict(scenario='materials', units='completed_bookings', policy='fixed-origin-recursive',
                evaluated_horizons=[1, 2, 3], forecast_horizon=months, models=models, split=split), rows


def price_values(part, ratios):
    base = part.base_price.to_numpy(dtype=float)
    return np.clip(np.asarray(ratios).reshape(-1), .85, 1.15)*base


def formula_values(part):
    maximum = part.groupby(['year', 'month']).total_bookings.transform('max').replace(0, 1)
    ratio = 1+.05*part.total_bookings/maximum-.03*part.first_visit_share
    return np.round(price_values(part, ratio), 2)


def compare_price(data, output, settings, hidden_sizes):
    from .contracts import validate_prepared
    from .training import split_data
    from .regressors import fit_candidates
    from .scenarios.price import predict
    frame = validate_prepared(data, 'price', True)
    parts = split_data(frame, 'price')
    raw = [part[list(PRICE.features)].values.astype(np.float32) for part in parts]
    mean, std = raw[0].mean(axis=0), raw[0].std(axis=0)
    std[std == 0] = 1
    arrays = [(values-mean)/std for values in raw]
    targets = [part.price_ratio.values.astype(np.float32).reshape(-1, 1) for part in parts]
    def validation_objective(state):
        ratios = Predictor(state).predict(arrays[1])
        return regression_metrics(parts[1].target_price.values, price_values(parts[1], ratios))['mse']
    states, _, details, _ = fit_candidates(arrays, targets, PRICE.features, settings,
                                          hidden_sizes or PRICE.hidden_sizes,
                                          validation_evaluator=validation_objective)
    # Task baseline: keep the current price, not mean price_ratio.
    states['baseline']['intercept'] = 1.0
    models, rows = {}, []
    for kind in ('baseline', 'linear', 'mlp', 'formula'):
        models[kind] = dict(training={k: v for k, v in details.get(kind, {'fit_seconds': 0, 'policy': 'direct rule'}).items() if k not in ('validation', 'test')})
        state = states.get(kind)
        if state is not None:
            state['feature_mean'], state['feature_std'] = mean.tolist(), std.tolist()
            predictor = Predictor(state)
        for label, part in zip(('validation', 'test'), parts[1:]):
            predicted = formula_values(part) if kind == 'formula' else price_values(part, predictor.predict(part[list(PRICE.features)].values))
            models[kind][label] = regression_metrics(part.target_price.values, predicted)
            if label == 'test':
                rows.extend(dict(model=kind, service_id=int(row.service_id), year=int(row.year), month=int(row.month),
                                 actual=float(row.target_price), prediction=float(value))
                            for row, value in zip(part.itertuples(), predicted))
        models[kind]['training']['validation_mse'] = models[kind]['validation']['mse']
        if kind in ('mlp', 'linear'):
            state['price_context'] = dict(task='formula-derived-price-recommendation',
                                          evaluation='monthly-rule-reconstruction', units='currency')
            state['metrics'] = models[kind]
            torch.save(state, output/f'price-{kind}.pth')
            if kind == 'mlp':
                torch.save(state, output/'price_model.pth')
            (output/f'price-{kind}-predictions.json').write_text(json.dumps(predict(frame, predictor), indent=2), encoding='utf-8')
    split = {label: dict(rows=len(part), start=f'{int(part.year.iloc[0])}-{int(part.month.iloc[0]):02d}',
                        end=f'{int(part.year.iloc[-1])}-{int(part.month.iloc[-1]):02d}')
             for label, part in zip(('train', 'validation', 'test'), parts)}
    return dict(scenario='price', units='currency', task='recommendation, not observed future price',
                evaluation='monthly-rule-reconstruction; clipping to 85–115%; no annual aggregation or unpopular-service discount',
                formula_note='rule reconstructed from rounded export features; cent-level differences from source labels are possible',
                models=models, split=split), rows


def compare_service(scenario, data_dir, output_dir, *, settings=None, hidden_sizes=None, months=3):
    from .storage import execute_run
    from .storage.specs import checkpoint_spec
    if scenario not in ('materials', 'price') or type(months) is not int or months not in (1, 2, 3):
        raise ValueError('invalid scenario or months')
    prefix = 'material' if scenario == 'materials' else 'price'
    def specs(stage, report):
        from .provenance import checkpoint_provenance
        provenance = checkpoint_provenance(data_dir, scenario)
        return [checkpoint_spec(stage/f'{prefix}-{kind}.pth', scenario, report['settings'], provenance=provenance) for kind in ('mlp', 'linear')]
    report, _, versions = execute_run(
        output_dir, f'compare-{scenario}', dict(data=str(data_dir), settings=asdict(settings) if settings else None,
                                               hidden_sizes=hidden_sizes, months=months),
        lambda stage, rid: _compare_service(scenario, data_dir, stage, settings=settings,
                                           hidden_sizes=hidden_sizes, months=months, run_id=rid), specs)
    report['version_ids'] = versions
    return report
