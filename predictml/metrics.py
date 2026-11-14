"""Original ratio metrics; zero-denominator policy retained until iteration 5."""
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
