"""Train, save and apply an independent model for a configured table schema."""
from dataclasses import asdict
from datetime import datetime, timezone
import uuid
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import torch
from .schema import TableConfig, numeric, dates
from .preprocessing import Preprocessor
from .io import read_table
from ..config import TrainingConfig
from ..training import fit_arrays
from ..inference import Predictor
from ..metrics import regression_metrics


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


class TabularModel:
    format_version = 1

    def __init__(self, schema, preprocessor, checkpoint, report):
        self.format_version = 1
        self.schema, self.preprocessor = schema, preprocessor
        self.checkpoint, self.report = checkpoint, report

    def predict(self, frame, model_kind=None):
        if frame.empty:
            raise ValueError('prediction: no rows')
        values = self.preprocessor.transform(frame)
        with torch.random.fork_rng(devices=[]):
            state = self.checkpoint if model_kind is None else getattr(self, 'checkpoints', {'mlp': self.checkpoint})[model_kind]
            result = Predictor(state).predict(values).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError('prediction: non-finite model output')
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        if Path(path).is_dir():
            from ..storage.bundles import load_payload
            _, model = load_payload(path)
            if not isinstance(model, cls):
                raise ValueError('expected tabular model bundle')
            return model
        model = joblib.load(path)
        if not isinstance(model, cls) or model.format_version != 1:
            raise ValueError('model: unsupported tabular bundle')
        if tuple(model.schema['features']) != model.preprocessor.features:
            raise ValueError('model: schema does not match preprocessor')
        if tuple(model.checkpoint['features']) != model.preprocessor.output_names:
            raise ValueError('model: transformed feature order mismatch')
        return model


def train_table(frame, config, settings=None, hidden_sizes=(32, 16), model_kind="mlp"):
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
    settings = settings or TrainingConfig(epochs=50)
    from ..regressors import fit_candidates
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
    model.checkpoints = states
    return model, table


def _train_file(path, output_dir, config, *, sheet=None, settings=None, hidden_sizes=(32, 16), model_kind="mlp", run_id=None):
    model, predictions = train_table(read_table(path, sheet), config, settings, hidden_sizes, model_kind)
    if run_id is not None:
        model.report['run_id'] = run_id
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output/'model.joblib')
    for name, state in model.checkpoints.items():
        variant = TabularModel(model.schema, model.preprocessor, state, {**model.report, 'selected_model': name})
        variant.save(output/f'model-{name}.joblib')
    for name, value in [('schema', model.schema), ('report', model.report)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    predictions.to_csv(output/'test_predictions.csv', index=False)
    from ..reporting import export_comparison
    rows = [dict(source_row=int(row.source_row), actual=float(row.actual),
                 prediction=float(getattr(row, f'prediction_{name}')), model=name)
            for row in predictions.itertuples() for name in model.checkpoints]
    export_comparison(output, model.report['models'], rows, model.report['split'],
                      {k: v for k, v in model.report.items() if k not in ('models', 'split')})
    return model


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


def train_file(path, output_dir, config, *, sheet=None, settings=None, hidden_sizes=(32, 16), model_kind='mlp'):
    from ..storage import execute_run
    from ..storage.specs import tabular_specs
    model, run_id, versions = execute_run(
        output_dir, 'train-tabular', dict(file=str(path), sheet=sheet, config=config,
                                        settings=asdict(settings) if settings else None, hidden_sizes=hidden_sizes, model_kind=model_kind),
        lambda stage, rid: _train_file(path, stage, config, sheet=sheet, settings=settings,
                                      hidden_sizes=hidden_sizes, model_kind=model_kind, run_id=rid), tabular_specs)
    model.storage = dict(run_id=run_id, version_ids=versions, store=str(Path(output_dir).resolve()))
    return model


def predict_file(path, model_path, output_path, *, sheet=None):
    from ..storage import execute_run
    destination = Path(output_path)
    if destination.suffix.lower() not in ('.csv', '.xlsx'):
        raise ValueError('prediction output must be CSV or XLSX')
    result, _, _ = execute_run(destination.parent, 'predict-tabular',
                              dict(file=str(path), model=str(model_path), output=destination.name, sheet=sheet),
                              lambda stage, rid: _predict_file(path, model_path, stage/destination.name, sheet=sheet))
    return result
