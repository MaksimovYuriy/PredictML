"""Adapters for the original ServiceFlow training and ratio contracts."""
import numpy as np
from ..evaluation import compute_metrics
from ..training import fit_prepared
from ..scenarios.demand import split_data as split_demand, baseline_metrics as demand_baselines
from ..scenarios.price import split_data as split_price


def split_data(df, scenario):
    if scenario == 'materials':
        return split_demand(df)
    if scenario == 'price':
        return split_price(df)
    raise ValueError('unknown scenario')


def baseline_metrics(test_df, targets, scenario, mae, features=None):
    if scenario == 'materials':
        return demand_baselines(test_df, targets, mae, features)
    mse, best, wmape = compute_metrics(targets, np.ones_like(targets))
    return dict(baseline_mse=mse, baseline_mae=best, baseline_wmape=wmape,
                mae_improvement_pct=(best-mae)/best*100 if best > 0 else 0.0)


def fit(df, scenario, model_config, training_config, **kwargs):
    return fit_prepared(split_data(df, scenario), model_config, training_config,
                        baseline_evaluator=lambda test, target, mae, features: baseline_metrics(
                            test, target, scenario, mae, features), **kwargs)
