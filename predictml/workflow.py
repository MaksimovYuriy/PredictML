"""User operations shared by CLI and UI; orchestrates computation and persistence."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import uuid
import numpy as np
import pandas as pd
import torch
from .config import (CONFIGS, PRICE, TrainingConfig, DEFAULT_HIDDEN_SIZES,
                     COMPARISON_TRAINING_DEFAULTS, INTERACTIVE_TRAINING_DEFAULTS)
from .data import read_table
from .models import Predictor
from .prediction import TabularModel
from .training import train_table
from .storage import Store, execute_run
from .storage.bundles import checkpoint_spec, tabular_specs, digest
from .evaluation import (regression_metrics,
                         checkpoint_provenance, merge_provenance, family, evaluate_control)
from .reporting import export_comparison, export_frames, metrics_table
from .updating import tabular_update, checkpoint_update
from .scenarios.demand import CONFIG, COLUMNS, load_monthly, fit_demand, forecast, validate_monthly
from .scenarios.materials import load_catalog, calculate
from .scenarios.price import price_values, formula_values


def _predict_file(path, model_path, output_path, *, sheet=None):
    frame = read_table(path, sheet)
    model = TabularModel.load(model_path)
    output = pd.DataFrame(dict(source_row=np.arange(len(frame)), prediction=model.predict(frame)))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == '.xlsx':
        output.to_excel(destination, index=False, engine='openpyxl')
    elif destination.suffix.lower() == '.csv':
        output.to_csv(destination, index=False)
    else:
        raise ValueError('prediction output must be CSV or XLSX')
    return output


def _train_file(path, output_dir, config, *, sheet=None, settings=None, hidden_sizes=DEFAULT_HIDDEN_SIZES, model_kind="mlp", run_id=None):
    model, predictions = train_table(read_table(path, sheet), config, settings, hidden_sizes, model_kind)
    if run_id is not None:
        model.report['run_id'] = run_id
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output/'model.joblib')
    for name, state in model.checkpoints.items():
        variant = TabularModel(model.schema, model.preprocessor, state, {**model.report, 'selected_model': name})
        variant.provenance = model.provenance
        variant.save(output/f'model-{name}.joblib')
    for name, value in [('schema', model.schema), ('report', model.report)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    predictions.to_csv(output/'test_predictions.csv', index=False)
    rows = [dict(source_row=int(row.source_row), actual=float(row.actual),
                 prediction=float(getattr(row, f'prediction_{name}')), model=name)
            for row in predictions.itertuples() for name in model.checkpoints]
    export_comparison(output, model.report['models'], rows, model.report['split'],
                      {k: v for k, v in model.report.items() if k not in ('models', 'split')})
    return model


def train_file(path, output_dir, config, *, sheet=None, settings=None, hidden_sizes=DEFAULT_HIDDEN_SIZES, model_kind='mlp'):
    model, run_id, versions = execute_run(
        output_dir, 'train-tabular', dict(file=str(path), sheet=sheet, config=config,
                                        settings=asdict(settings) if settings else None, hidden_sizes=hidden_sizes, model_kind=model_kind),
        lambda stage, rid: _train_file(path, stage, config, sheet=sheet, settings=settings,
                                      hidden_sizes=hidden_sizes, model_kind=model_kind, run_id=rid), tabular_specs)
    model.storage = dict(run_id=run_id, version_ids=versions, store=str(Path(output_dir).resolve()))
    return model


def predict_file(path, model_path, output_path, *, sheet=None):
    destination = Path(output_path)
    if destination.suffix.lower() not in ('.csv', '.xlsx'):
        raise ValueError('prediction output must be CSV or XLSX')
    result, _, _ = execute_run(destination.parent, 'predict-tabular',
                              dict(file=str(path), model=str(model_path), output=destination.name, sheet=sheet),
                              lambda stage, rid: _predict_file(path, model_path, stage/destination.name, sheet=sheet))
    return result


def _run(scenario, data_dir, output_dir, *, train=True, model=None,
        months=3, epochs=None, mode="demand-v2"):
    if scenario not in ('materials', 'price'):
        raise ValueError('scenario must be materials or price')
    if type(months) is not int or months not in (1, 2, 3):
        raise ValueError('months must be 1, 2 or 3')
    if mode not in ('demand-v2', 'legacy-parity'):
        raise ValueError('mode: expected demand-v2 or legacy-parity')
    if scenario == 'materials' and mode == 'demand-v2':
        return run_demand(data_dir, output_dir, train=train, model=model, months=months, epochs=epochs)
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    prefix = 'material' if scenario == 'materials' else 'price'
    csv = data / ('material_forecast.csv' if scenario == 'materials' else 'price_analysis.csv')
    required = [csv]
    if scenario == 'materials':
        required += [data / 'materials.json', data / 'service_materials.json']
    for path in required:
        if not path.is_file():
            raise ValueError(f'Missing input file: {path}')
    checkpoint = Path(model).resolve() if model else output / f'{prefix}_model.pth'
    if not train and not checkpoint.is_file():
        raise ValueError(f'Missing model: {checkpoint}')
    if train and model:
        raise ValueError('model is only accepted for prediction without training')
    if epochs is not None and epochs < 50:
        raise ValueError('Original trainer requires at least 50 epochs')
    from .scenarios.serviceflow import validate_prepared
    frame = validate_prepared(data, scenario, train)
    output.mkdir(parents=True, exist_ok=True)
    from .compatibility.legacy import fit
    from .compatibility import materials
    from .scenarios import price

    config = CONFIGS[scenario]
    if train:
        settings = TrainingConfig(epochs=config.epochs if epochs is None else epochs)
        state = fit(frame, scenario, config, settings)
        torch.save(state, checkpoint)
        (output / f'{prefix}_metrics.json').write_text(
            json.dumps(state['metrics'], indent=2, ensure_ascii=False), encoding='utf-8')
        predictor = Predictor(state)
    else:
        predictor = Predictor.load(checkpoint)
    if scenario == 'materials':
        norms = json.loads((data/'service_materials.json').read_text(encoding='utf-8'))
        records = json.loads((data/'materials.json').read_text(encoding='utf-8'))
        result = materials.predict(frame, norms, records, predictor, months)
    else:
        result = price.predict(frame, predictor)
    destination = output/f'{prefix}_predictions.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def run(scenario, data_dir, output_dir, *, train=True, model=None, months=3, epochs=None, mode='demand-v2'):
    if scenario not in ('materials', 'price'):
        raise ValueError('scenario must be materials or price')
    if type(months) is not int or months not in (1, 2, 3):
        raise ValueError('months must be 1, 2 or 3')
    if mode not in ('demand-v2', 'legacy-parity'):
        raise ValueError('invalid mode')
    prefix = 'material' if scenario == 'materials' else 'price'
    selected_model = model if model is not None or train else Path(output_dir)/f'{prefix}_model.pth'
    def specs(stage, result):
        settings = TrainingConfig(epochs=CONFIGS[scenario].epochs if epochs is None else epochs)
        return [checkpoint_spec(stage/f'{prefix}_model.pth', scenario, asdict(settings), mode=mode,
                                provenance=checkpoint_provenance(data_dir, scenario, mode))]
    result, _, _ = execute_run(output_dir, ('train-' if train else 'predict-')+scenario,
                              dict(data=str(data_dir), model=str(selected_model) if selected_model else None,
                                   months=months, epochs=epochs, mode=mode),
                              lambda stage, rid: _run(scenario, data_dir, stage, train=train, model=selected_model,
                                                      months=months, epochs=epochs, mode=mode), specs if train else None)
    return result


def run_demand(data_dir, output_dir, *, train=True, model=None, months=3, epochs=None):
    data, output = Path(data_dir), Path(output_dir)
    if train and model:
        raise ValueError('model is only accepted for prediction without training')
    if train:
        settings = TrainingConfig(epochs=CONFIG.epochs if epochs is None else epochs)
        frame, metadata = load_monthly(data)
        materials, norms = load_catalog(data, set(frame.service_id))
        state = fit_demand(frame, metadata, settings)
        predictor = Predictor(state)
    else:
        if model is None:
            model = output/'material_model.pth'
        predictor = Predictor.load(model)
        context = predictor.demand_context
        if not isinstance(context, dict) or context.get('contract_version') != 2 or context.get('target') != 'total_bookings' or context.get('frequency') != 'MS':
            raise ValueError('model: demand-v2 context required; old ratio checkpoints require --mode legacy-parity')
        stored = validate_monthly(pd.DataFrame(context['history']))
        frame = stored
        if (data/'monthly_demand.csv').exists():
            frame, metadata = load_monthly(data)
            if set(frame.service_id) != set(stored.service_id):
                raise ValueError('monthly_demand.service_id: model requires the same service set')
            if frame.ym.max() < stored.ym.max():
                raise ValueError('monthly_demand: cannot forecast before saved observation boundary')
            overlap = frame[frame.ym <= stored.ym.max()][COLUMNS].reset_index(drop=True)
            if not overlap.astype(int).equals(stored[COLUMNS].astype(int).reset_index(drop=True)):
                raise ValueError('monthly_demand: saved history must be unchanged; append complete months or train a new model')
        materials, norms = load_catalog(data, set(frame.service_id))
    demand = forecast(frame, predictor, months)
    result = calculate(demand, materials, norms)
    output.mkdir(parents=True, exist_ok=True)
    if train:
        torch.save(state, output/'material_model.pth')
        (output/'material_metrics.json').write_text(json.dumps(state['metrics'], indent=2, allow_nan=False), encoding='utf-8')
        report = state['metrics']['recursive_test']
        rows = [{**row, 'model': name, 'origin': report['origin']} for name, details in report['models'].items() for row in details['rows']]
        pd.DataFrame(rows).to_csv(output/'demand_test_predictions.csv', index=False)
    (output/'demand_predictions.json').write_text(json.dumps(demand, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(demand).to_csv(output/'demand_predictions.csv', index=False)
    (output/'material_predictions.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


def _compare_service(scenario, data_dir, output_dir, *, settings=None, hidden_sizes=None, months=3, run_id=None):
    settings = settings or COMPARISON_TRAINING_DEFAULTS
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


def compare_price(data, output, settings, hidden_sizes):
    from .scenarios.serviceflow import validate_prepared
    from .scenarios.price import split_data
    from .training import fit_candidates
    from .scenarios.price import predict
    frame = validate_prepared(data, 'price', True)
    parts = split_data(frame)
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
    if scenario not in ('materials', 'price') or type(months) is not int or months not in (1, 2, 3):
        raise ValueError('invalid scenario or months')
    prefix = 'material' if scenario == 'materials' else 'price'
    def specs(stage, report):
        provenance = checkpoint_provenance(data_dir, scenario)
        return [checkpoint_spec(stage/f'{prefix}-{kind}.pth', scenario, report['settings'], provenance=provenance) for kind in ('mlp', 'linear')]
    report, _, versions = execute_run(
        output_dir, f'compare-{scenario}', dict(data=str(data_dir), settings=asdict(settings) if settings else None,
                                               hidden_sizes=hidden_sizes, months=months),
        lambda stage, rid: _compare_service(scenario, data_dir, stage, settings=settings,
                                           hidden_sizes=hidden_sizes, months=months, run_id=rid), specs)
    report['version_ids'] = versions
    return report


def publish_files(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in Path(source).iterdir():
        if not path.is_file():
            continue
        temporary = destination/f'.export-{uuid.uuid4()}-{path.name}'
        try:
            shutil.copyfile(path, temporary)
            os.replace(temporary, destination/path.name)
        finally:
            temporary.unlink(missing_ok=True)


def predict_version(root, version_id, data, output_dir=None, *, months=3, sheet=None, export_format='both'):
    store = Store(root)
    if output_dir is not None:
        store.check_destination(output_dir)
    def worker(stage, run_id):
        bundle, manifest, payload = store.load(version_id)
        task = manifest['task']
        if task == 'tabular':
            frame = read_table(data, sheet)
            frames = {'predictions': pd.DataFrame(dict(source_row=range(len(frame)), prediction=payload.predict(frame)))}
        elif task == 'materials':
            if manifest['temporal'].get('demand_context'):
                result = run_demand(data, stage, train=False, model=bundle, months=months)
            else:
                from .scenarios.serviceflow import validate_prepared
                from .compatibility.materials import predict
                source = Path(data)
                result = predict(validate_prepared(source, 'materials', False),
                                 json.loads((source/'service_materials.json').read_text()),
                                 json.loads((source/'materials.json').read_text()), Predictor(payload), months)
                (stage/'material_predictions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            frames = {
                'materials': pd.DataFrame([{k: v for k, v in row.items() if k != 'months'} for row in result]),
                'material_months': pd.DataFrame([dict(material_id=row['material_id'], **month) for row in result for month in row['months']])}
            if (stage/'demand_predictions.csv').exists():
                frames['demand'] = pd.read_csv(stage/'demand_predictions.csv')
        elif task == 'price':
            from .scenarios.serviceflow import validate_prepared
            from .scenarios.price import predict
            result = predict(validate_prepared(data, 'price', False), Predictor(payload))
            frames = {'prices': pd.DataFrame(result)}
            (stage/'price_predictions.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        else:
            raise ValueError(f'unsupported model task {task}')
        export_frames(stage, frames, manifest['metrics'], export_format)
        (stage/'prediction.json').write_text(json.dumps(dict(run_id=run_id, version_id=version_id,
                                                             model_created_at=manifest['created_at'], task=task,
                                                             model_kind=manifest['model_kind']), indent=2), encoding='utf-8')
        return frames
    if type(months) is not int or months not in (1, 2, 3):
        raise ValueError('months must be 1, 2 or 3')
    _, run_id, _ = execute_run(root, 'predict-version', dict(data=str(data), months=months, sheet=sheet, format=export_format),
                              worker, input_version=version_id, publish=False)
    canonical = store.root/'runs'/run_id
    if output_dir is not None:
        publish_files(canonical, output_dir)
    return dict(run_id=run_id, version_id=version_id, output_path=str(canonical))


def export_metrics(root, version_id, destination):
    destination = Path(destination)
    if destination.suffix.lower() not in ('.csv', '.xlsx'):
        raise ValueError('metrics export must be CSV or XLSX')
    store = Store(root)
    store.check_destination(destination)
    def worker(stage, run_id):
        _, _, manifest = store.version(version_id)
        table = metrics_table(manifest['metrics'])
        if destination.suffix.lower() == '.csv':
            table.to_csv(stage/destination.name, index=False)
        else:
            table.to_excel(stage/destination.name, index=False, engine='openpyxl')
    _, run_id, _ = execute_run(root, 'export-metrics', dict(destination=str(destination)), worker,
                              input_version=version_id, publish=False)
    source = store.root/'runs'/run_id/destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent/f'.export-{uuid.uuid4()}'
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def update_version(root, parent_id, data, *, operation='retrain', settings=None, config=None, hidden_sizes=None, sheet=None):
    if operation not in ('retrain', 'fine-tune'):
        raise ValueError('operation must be retrain or fine-tune')
    settings = settings or INTERACTIVE_TRAINING_DEFAULTS
    store = Store(root)
    metadata = {}
    def worker(stage, run_id):
        _, manifest, parent = store.load(parent_id)
        if operation == 'fine-tune' and manifest['model_kind'] != 'mlp':
            raise ValueError('fine-tune is supported only for MLP')
        if manifest['task'] == 'tabular':
            model = tabular_update(parent, read_table(data, sheet), settings, operation, config, hidden_sizes)
            compatible = family(model.schema) == family(manifest['schema'])
            model.provenance = merge_provenance(model.provenance, manifest.get('provenance'), operation == 'fine-tune')
            model.report.update(run_id=run_id, update_operation=operation, parent_version=parent_id,
                                optimizer_policy='fresh Adam' if operation == 'fine-tune' else 'from scratch')
            name = model.report['selected_model']
            model.save(stage/f'model-{name}.joblib')
            spec = next(tabular_specs(stage, model))
        else:
            if config is not None:
                raise ValueError('scenario schema is fixed; use train-table for a different target/features')
            state, provenance = checkpoint_update(parent, manifest['task'], data, settings, operation, hidden_sizes)
            torch.save(state, stage/'model.pth')
            spec = checkpoint_spec(stage/'model.pth', manifest['task'], state['metrics']['training']['settings'],
                                   provenance=merge_provenance(provenance, manifest.get('provenance'), operation == 'fine-tune'))
            compatible = True
        spec.update(parent_id=parent_id if compatible else None, origin_version=parent_id,
                    update_operation=operation if compatible else 'new-model',
                    optimizer_policy='fresh Adam; saved weights' if operation == 'fine-tune' else 'from scratch')
        metadata['spec'] = spec
        (stage/'update.json').write_text(json.dumps({k: v for k, v in spec.items() if k != 'source'}, indent=2, allow_nan=False), encoding='utf-8')
        return dict(parent_id=spec['parent_id'], operation=spec['update_operation'])
    result, run_id, versions = execute_run(root, operation, dict(parent_id=parent_id, data=str(data), settings=asdict(settings),
                                                               config=config, hidden_sizes=hidden_sizes, sheet=sheet), worker,
                                         lambda stage, result: [metadata['spec']], input_version=parent_id, publish=False)
    return {**result, 'run_id': run_id, 'version_id': versions[0]}


def compare_versions(root, version_ids, control, *, sheet=None, history=None):
    if len(version_ids) < 2 or len(set(version_ids)) != len(version_ids):
        raise ValueError('select at least two distinct version IDs')
    store = Store(root)
    def worker(stage, run_id):
        loaded = [store.load(version) for version in version_ids]
        manifests, payloads = [item[1] for item in loaded], [item[2] for item in loaded]
        frame, models, rows, conditions, provenance, observed = evaluate_control(
            manifests, payloads, read_table(control, sheet), version_ids,
            history=read_table(history) if history is not None and manifests[0]['task'] == 'materials' else None)
        if observed is not None:
            observed[COLUMNS].to_csv(stage/'history.csv', index=False)
        report = dict(run_id=run_id, version_ids=version_ids,
                      versions=[{k: m.get(k) for k in ('version_id','model_kind','parent_id','created_at','settings','architecture','update_operation')} for m in manifests],
                      control_file_sha256=digest(control),
                      control_rows=len(frame), control_provenance=provenance, conditions=conditions, models=models,
                      independence='checked against all supplied training/validation/test observations and inherited weights',
                      policy='evaluation only; no automatic selection or learning from control')
        frame.to_csv(stage/'control.csv', index=False)
        (stage/'control_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')
        export_comparison(stage, models, rows, conditions, {k:v for k,v in report.items() if k!='models'})
        export_frames(stage, {'predictions':pd.DataFrame(rows)}, models)
        return report
    report, run_id, _ = execute_run(root,'compare-versions',dict(version_ids=version_ids, control=str(control),
                                                               history=str(history) if history else None,sheet=sheet),worker,publish=False)
    return {**report,'output_path':str(store.root/'runs'/run_id)}
