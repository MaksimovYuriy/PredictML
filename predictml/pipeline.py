"""Explicit paths around the original ServiceFlow algorithms.

Legacy implementations are retained for parity, including their documented
methodological limitations. Each working directory belongs to one run.
"""
from importlib import import_module
from pathlib import Path
import json


def run(scenario, data_dir, output_dir, *, train=True, model=None,
        months=3, epochs=None):
    if scenario not in ('materials', 'price'):
        raise ValueError('scenario must be materials or price')
    if months not in (1, 2, 3):
        raise ValueError('months must be 1, 2 or 3')
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
    from .contracts import validate_prepared
    validate_prepared(data, scenario, train)
    output.mkdir(parents=True, exist_ok=True)
    # Modules are imported per scenario; no Rails or source-repository paths.
    trainer = import_module(f'predictml.legacy.train_{scenario}')
    predictor = import_module(f'predictml.legacy.predict_{scenario}')
    for module in (trainer, predictor):
        module.CSV_PATH = str(csv)
        module.MODEL_PATH = str(checkpoint)
    trainer.METRICS_PATH = str(output / f'{prefix}_metrics.json')
    predictor.OUTPUT_PATH = str(output / f'{prefix}_predictions.json')
    if scenario == 'materials':
        predictor.SERVICE_MATERIALS_PATH = str(data / 'service_materials.json')
        predictor.MATERIALS_PATH = str(data / 'materials.json')
        predictor.PREDICT_MONTHS = months
    original_epochs = trainer.EPOCHS
    try:
        if epochs is not None:
            trainer.EPOCHS = epochs
        if train:
            trainer.train()
        predictor.predict()
    finally:
        trainer.EPOCHS = original_epochs
    return json.loads(Path(predictor.OUTPUT_PATH).read_text())
