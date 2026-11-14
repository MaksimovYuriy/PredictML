"""Regression metrics; historical ratio helper retained for legacy parity."""
import numpy as np


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    error = y_true - y_pred
    mse = float((error ** 2).mean())
    mae = float(np.abs(error).mean())
    denominator = float(np.abs(y_true).sum())
    wmape = float(np.abs(error).sum() / denominator * 100) if denominator > 0 else 0.0
    return mse, mae, wmape


def regression_metrics(y_true, y_pred):
    """Original units; undefined relative metrics are represented by None."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if not len(actual) or actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('metrics: expected nonempty finite actual/prediction arrays of equal length')
    error = actual-predicted
    mse = float(np.mean(error**2))
    denominator = float(np.abs(actual).sum())
    variance = float(((actual-actual.mean())**2).sum())
    return dict(mae=float(np.mean(np.abs(error))), mse=mse, rmse=float(np.sqrt(mse)),
                wmape=float(np.abs(error).sum()/denominator*100) if denominator else None,
                r2=float(1-(error**2).sum()/variance) if len(actual) > 1 and variance > 0 else None)
