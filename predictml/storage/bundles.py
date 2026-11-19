"""Self-contained checked model directories; load only trusted pickle payloads."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from ..config import BUNDLE_PACKAGES as PACKAGES


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    from .. import __version__
    return dict(python=platform.python_version(), predictml=__version__, packages=packages)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_bundle(directory, source, manifest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    name = 'model.joblib' if manifest['payload_format'] == 'tabular' else 'model.pth'
    shutil.copyfile(source, directory/name)
    value = {**manifest, 'bundle_format': 1, 'environment': environment(),
             'payload': name, 'files': {name: dict(sha256=digest(directory/name), size=(directory/name).stat().st_size)}}
    (directory/'manifest.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return value


def verify_bundle(directory, *, manifest_hash=None, compatible=True):
    directory = Path(directory).resolve()
    try:
        path = directory/'manifest.json'
        if manifest_hash is not None and digest(path) != manifest_hash:
            raise ValueError('manifest checksum mismatch')
        value = json.loads(path.read_text(encoding='utf-8'))
        required = {'bundle_format', 'version_id', 'run_id', 'created_at', 'task', 'model_kind',
                    'schema', 'settings', 'metrics', 'temporal', 'environment', 'payload', 'files', 'payload_format',
                    'seed', 'architecture', 'preprocessing'}
        if not isinstance(value, dict) or required-set(value) or value['bundle_format'] != 1:
            raise ValueError('unsupported or incomplete manifest')
        if value['payload_format'] not in ('tabular', 'checkpoint'):
            raise ValueError('unsupported payload format')
        if value['task'] not in ('tabular', 'materials', 'price') or (value['task'] == 'tabular') != (value['payload_format'] == 'tabular'):
            raise ValueError('incompatible task/payload format')
        if any(not isinstance(value[key], dict) for key in ('settings', 'metrics', 'temporal', 'architecture', 'preprocessing', 'environment')):
            raise ValueError('invalid metadata objects')
        expected = 'model.joblib' if value['payload_format'] == 'tabular' else 'model.pth'
        if value['payload'] != expected or set(value['files']) != {expected}:
            raise ValueError('invalid payload inventory')
        if not isinstance(value['schema'], dict) or not value['schema'].get('target') or not value['schema'].get('features'):
            raise ValueError('missing target or feature schema')
        features = value['schema']['features']
        if not isinstance(features, list) or any(not isinstance(name, str) for name in features) or len(features) != len(set(features)):
            raise ValueError('invalid feature names/order')
        types = value['schema'].get('types')
        if not isinstance(types, dict) or set(types) != set(features) or any(t not in ('numeric', 'categorical', 'datetime') for t in types.values()):
            raise ValueError('invalid feature type schema')
        if not isinstance(value['schema']['target'], str):
            raise ValueError('invalid target')
        file = directory/expected
        if file.is_symlink() or not file.is_file():
            raise ValueError('missing payload or symlink')
        info = value['files'][expected]
        if file.stat().st_size != info['size'] or digest(file) != info['sha256']:
            raise ValueError('payload checksum/size mismatch')
        if compatible:
            recorded, current = value['environment'], environment()
            if recorded['python'].split('.')[:2] != current['python'].split('.')[:2]:
                raise ValueError('incompatible Python major/minor version')
            # Conservative, documented serialization policy: exact versions.
            for name in ('numpy', 'pandas', 'torch', 'scikit-learn', 'scipy', 'joblib'):
                if recorded['packages'].get(name) != current['packages'].get(name):
                    raise ValueError(f'incompatible dependency {name}: saved {recorded["packages"].get(name)}, installed {current["packages"].get(name)}')
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'bundle {directory.name}: {exc}') from exc


def load_payload(directory, *, manifest_hash=None):
    manifest = verify_bundle(directory, manifest_hash=manifest_hash)
    path = Path(directory)/manifest['payload']
    try:
        if manifest['payload_format'] == 'tabular':
            from ..prediction import TabularModel
            payload = TabularModel.load(path)
            if payload.schema != manifest['schema'] or payload.checkpoint.get('model_kind', 'mlp') != manifest['model_kind']:
                raise ValueError('payload schema/model kind differs from manifest')
            if payload.schema['types'] != payload.preprocessor.types or list(payload.preprocessor.output_names) != manifest['preprocessing'].get('output_features'):
                raise ValueError('payload preprocessor differs from metadata')
            state = payload.checkpoint
        else:
            import torch
            from ..models import Predictor
            payload = torch.load(path, map_location='cpu', weights_only=False)
            Predictor(payload)
            if list(payload['features']) != manifest['schema']['features'] or payload.get('model_kind', 'mlp') != manifest['model_kind']:
                raise ValueError('payload schema/model kind differs from manifest')
            if payload.get('demand_context') != manifest['temporal'].get('demand_context'):
                raise ValueError('payload temporal context differs from manifest')
            from ..config import PRICE, MATERIALS
            from ..scenarios.demand import FEATURES
            expected = PRICE.features if manifest['task'] == 'price' else FEATURES if payload.get('demand_context') else MATERIALS.features
            expected_target = 'price_ratio' if manifest['task'] == 'price' else 'total_bookings' if payload.get('demand_context') else 'booking_ratio'
            if manifest['schema']['target'] != expected_target or any(t != 'numeric' for t in manifest['schema']['types'].values()):
                raise ValueError('payload is incompatible with scenario target/types')
            if tuple(payload['features']) != tuple(expected):
                raise ValueError('payload is incompatible with scenario feature schema')
            state = payload
            preprocessing = manifest['preprocessing']
            if state['feature_mean'] != preprocessing.get('mean') or state['feature_std'] != preprocessing.get('std'):
                raise ValueError('payload normalization differs from metadata')
        architecture = manifest['architecture']
        if state['input_size'] != architecture.get('input_size') or state.get('hidden_sizes') != architecture.get('hidden_sizes') or state.get('dropout') != architecture.get('dropout'):
            raise ValueError('payload architecture differs from metadata')
        return manifest, payload
    except Exception as exc:
        raise ValueError(f'bundle payload could not be loaded: {exc}') from exc


def checkpoint_spec(source, task, settings, *, mode='demand-v2', provenance=None):
    import torch
    state = torch.load(source, map_location='cpu', weights_only=False)
    context = state.get('demand_context')
    target = 'price_ratio' if task == 'price' else 'total_bookings' if context else 'booking_ratio'
    features = list(state['features'])
    return dict(source=source, payload_format='checkpoint', task=task,
                model_kind=state.get('model_kind', 'mlp'),
                schema=dict(features=features, types={name: 'numeric' for name in features}, target=target,
                            target_units='completed_bookings' if context else 'ratio',
                            original_input=['service_id', 'year', 'month', 'total_bookings'] if context else None),
                settings=settings, seed=settings.get('seed'), metrics=state.get('metrics', {}), provenance=provenance,
                architecture=dict(input_size=state['input_size'], hidden_sizes=state.get('hidden_sizes'), dropout=state.get('dropout')),
                preprocessing=dict(kind='frozen-standardization', mean=state['feature_mean'], std=state['feature_std']),
                temporal=dict(mode=mode, demand_context=context, price_context=state.get('price_context')))


def tabular_specs(stage, model):
    for name, state in model.checkpoints.items():
        yield dict(source=stage/f'model-{name}.joblib', payload_format='tabular', task='tabular', model_kind=name,
                   schema=model.schema, settings=model.report['settings'], seed=model.report['seed'], provenance=getattr(model, 'provenance', None),
                   metrics=model.report['models'][name], temporal=dict(configuration=model.schema['temporal'], split=model.report['split']),
                   architecture=dict(input_size=state['input_size'], hidden_sizes=state.get('hidden_sizes'), dropout=state.get('dropout')),
                   preprocessing=dict(kind='fitted-sklearn-pipeline', output_features=list(model.preprocessor.output_names)))


def save_table_model(model, path):
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_table_model(path):
    from ..compatibility.serialization import load_joblib
    from ..prediction import TabularModel
    if Path(path).is_dir():
        _, model = load_payload(path)
        if not isinstance(model, TabularModel):
            raise ValueError('expected tabular model bundle')
        return model
    model = load_joblib(path)
    if not isinstance(model, TabularModel) or model.format_version != 1:
        raise ValueError('model: unsupported tabular bundle')
    if tuple(model.schema['features']) != model.preprocessor.features:
        raise ValueError('model: schema does not match preprocessor')
    if tuple(model.checkpoint['features']) != model.preprocessor.output_names:
        raise ValueError('model: transformed feature order mismatch')
    return model
