"""Standalone orchestration with explicit inputs and per-call model state."""
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
    frame = validate_prepared(data, scenario, train)
    output.mkdir(parents=True, exist_ok=True)
    from .config import CONFIGS, TrainingConfig
    from .training import fit
    from .inference import Predictor
    from .scenarios import materials, price
    import torch

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
