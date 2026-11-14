"""Metadata adapters for the existing model formats."""
from pathlib import Path
import torch


def checkpoint_spec(source, task, settings, *, mode='demand-v2'):
    state = torch.load(source, map_location='cpu', weights_only=False)
    context = state.get('demand_context')
    target = 'price_ratio' if task == 'price' else 'total_bookings' if context else 'booking_ratio'
    features = list(state['features'])
    return dict(source=source, payload_format='checkpoint', task=task,
                model_kind=state.get('model_kind', 'mlp'),
                schema=dict(features=features, types={name: 'numeric' for name in features}, target=target,
                            target_units='completed_bookings' if context else 'ratio',
                            original_input=['service_id', 'year', 'month', 'total_bookings'] if context else None),
                settings=settings, seed=settings.get('seed'), metrics=state.get('metrics', {}),
                architecture=dict(input_size=state['input_size'], hidden_sizes=state.get('hidden_sizes'), dropout=state.get('dropout')),
                preprocessing=dict(kind='frozen-standardization', mean=state['feature_mean'], std=state['feature_std']),
                temporal=dict(mode=mode, demand_context=context, price_context=state.get('price_context')))


def tabular_specs(stage, model):
    for name, state in model.checkpoints.items():
        yield dict(source=stage/f'model-{name}.joblib', payload_format='tabular', task='tabular', model_kind=name,
                   schema=model.schema, settings=model.report['settings'], seed=model.report['seed'],
                   metrics=model.report['models'][name], temporal=dict(configuration=model.schema['temporal'], split=model.report['split']),
                   architecture=dict(input_size=state['input_size'], hidden_sizes=state.get('hidden_sizes'), dropout=state.get('dropout')),
                   preprocessing=dict(kind='fitted-sklearn-pipeline', output_features=list(model.preprocessor.output_names)))
