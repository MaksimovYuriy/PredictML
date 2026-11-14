"""Prediction and export by immutable version ID."""
import json
import os
from pathlib import Path
import shutil
import uuid
import pandas as pd
from .store import Store, execute_run


def metrics_table(metrics):
    rows = []
    def visit(value, prefix=''):
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, f'{prefix}.{name}' if prefix else name)
        elif not isinstance(value, (list, tuple)):
            rows.append(dict(field=prefix, value=value))
    visit(metrics)
    return pd.DataFrame(rows, columns=['field', 'value'])


def export_frames(directory, frames, metrics, export_format='both'):
    if export_format not in ('csv', 'xlsx', 'both'):
        raise ValueError('format must be csv, xlsx or both')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = {**frames, 'metrics': metrics_table(metrics)}
    if export_format in ('csv', 'both'):
        for name, frame in frames.items():
            frame.to_csv(directory/f'{name}.csv', index=False)
    if export_format in ('xlsx', 'both'):
        with pd.ExcelWriter(directory/'results.xlsx', engine='openpyxl') as book:
            for name, frame in frames.items():
                frame.to_excel(book, sheet_name=name, index=False)


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
            from ..tabular.io import read_table
            frame = read_table(data, sheet)
            frames = {'predictions': pd.DataFrame(dict(source_row=range(len(frame)), prediction=payload.predict(frame)))}
        elif task == 'materials':
            from ..demand_pipeline import run_demand
            if manifest['temporal'].get('demand_context'):
                result = run_demand(data, stage, train=False, model=bundle, months=months)
            else:
                from ..contracts import validate_prepared
                from ..scenarios.materials import predict
                from ..inference import Predictor
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
            from ..contracts import validate_prepared
            from ..scenarios.price import predict
            from ..inference import Predictor
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
