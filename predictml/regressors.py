"""Common estimators and validation-only selection, independent of scenarios."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from .training import fit_arrays
from .inference import Predictor
from .metrics import regression_metrics


def fit_estimator(x_train, y_train, x_val, y_val, x_test, kind, hidden_sizes,
                  settings, features, mean, std, *, validation_evaluator=None, seed=None):
    started = perf_counter()
    log = {}
    state = dict(input_size=x_train.shape[1], features=list(features),
                 feature_mean=np.asarray(mean).tolist(), feature_std=np.asarray(std).tolist(), model_kind=kind)
    if kind == 'mlp':
        if not hidden_sizes or any(type(n) is not int or n < 1 for n in hidden_sizes):
            raise ValueError('hidden_sizes must contain positive integers')
        selected_seed = settings.seed if settings.seed is not None else seed
        if selected_seed is None:
            weights, best_loss, predicted = fit_arrays(
                x_train, y_train, x_val, y_val, x_test, hidden_sizes, settings,
                validation_evaluator=validation_evaluator, mean=mean, std=std, features=list(features), training_log=log)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(selected_seed)
                weights, best_loss, predicted = fit_arrays(
                    x_train, y_train, x_val, y_val, x_test, hidden_sizes, settings,
                    validation_evaluator=validation_evaluator, mean=mean, std=std, features=list(features), training_log=log)
        state.update(model_state=weights, hidden_sizes=list(hidden_sizes), dropout=settings.dropout)
    elif kind in ('linear', 'baseline'):
        if kind == 'linear':
            model = LinearRegression().fit(x_train.astype(float), y_train.reshape(-1).astype(float))
            coefficient, intercept = model.coef_, float(model.intercept_)
        else:
            coefficient, intercept = np.zeros(x_train.shape[1]), float(y_train.mean())
        state.update(coefficients=coefficient.tolist(), intercept=intercept)
        predicted = (x_test @ coefficient+intercept).reshape(-1, 1)
        val = (x_val @ coefficient+intercept).reshape(-1, 1)
        best_loss = validation_evaluator(state) if validation_evaluator else float(np.mean((val-y_val)**2))
    else:
        raise ValueError('model_kind must be mlp, linear or baseline')
    log.update(validation_mse=float(best_loss), fit_seconds=perf_counter()-started,
               settings=asdict(settings), hidden_sizes=list(hidden_sizes) if kind == 'mlp' else None)
    return state, predicted, log


def fit_candidates(arrays, targets, features, settings, hidden_sizes, *, seed=42, validation_evaluator=None):
    states, predictions, details = {}, {}, {}
    for kind in ('baseline', 'linear', 'mlp'):
        state, predicted, log = fit_estimator(*[arrays[0], targets[0], arrays[1], targets[1], arrays[2]],
                                            kind, hidden_sizes, settings, features,
                                            np.zeros(arrays[0].shape[1]), np.ones(arrays[0].shape[1]), seed=seed,
                                            validation_evaluator=validation_evaluator)
        with torch.random.fork_rng(devices=[]):
            predictor = Predictor(state)
            validation = predictor.predict(arrays[1]).reshape(-1)
        details[kind] = {**log, 'validation': regression_metrics(targets[1].reshape(-1), validation),
                         'test': regression_metrics(targets[2].reshape(-1), predicted.reshape(-1))}
        states[kind], predictions[kind] = state, predicted.reshape(-1)
    # The test is never used to choose the recommended model.
    selected = min(details, key=lambda name: details[name]['validation']['mse'])
    return states, predictions, details, selected
