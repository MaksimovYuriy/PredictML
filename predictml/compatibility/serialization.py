"""Resolve original class paths when reading pre-refactor joblib models."""
from importlib import import_module
import joblib
from joblib.numpy_pickle import NumpyUnpickler, _validate_fileobject_and_memmap

# Keep joblib's format-specific compatibility code here, outside the learning path.
CLASS_PATHS = {
    ('predictml.tabular.application', 'TabularModel'): ('predictml.prediction', 'TabularModel'),
    ('predictml.tabular.preprocessing', 'Preprocessor'): ('predictml.preprocessing', 'Preprocessor'),
    ('predictml.tabular.schema', 'TableConfig'): ('predictml.data', 'TableConfig'),
    ('predictml.models.mlp', 'MLP'): ('predictml.models', 'MLP'),
}


class LegacyUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        destination = CLASS_PATHS.get((module, name))
        if destination is not None:
            return getattr(import_module(destination[0]), destination[1])
        return super().find_class(module, name)


def load_joblib(path):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        if exc.name not in {module for module, _ in CLASS_PATHS} | {'predictml.tabular'}:
            raise
    with open(path, 'rb') as source:
        with _validate_fileobject_and_memmap(source, str(path), None) as (stream, _):
            return LegacyUnpickler(str(path), stream, ensure_native_byte_order=True).load()
