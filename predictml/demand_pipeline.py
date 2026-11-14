"""Orchestration for the corrected monthly-demand contract."""
import json
from pathlib import Path
import pandas as pd
import torch
from .config import TrainingConfig
from .demand import CONFIG, COLUMNS, load_monthly, fit_demand, forecast, validate_monthly
from .inference import Predictor
from .purchasing import load_catalog, calculate


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
