"""Pure evaluation helpers used by model workflows and automated tests."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.size == 0:
        raise ValueError("Evaluation data must contain at least one observation.")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have the same length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation values must all be finite.")
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def ensure_group_disjoint(groups, train_indices, test_indices):
    groups = np.asarray(groups)
    train_groups = set(groups[np.asarray(train_indices, dtype=int)])
    test_groups = set(groups[np.asarray(test_indices, dtype=int)])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        preview = ", ".join(sorted(map(str, overlap))[:5])
        raise ValueError(f"Train/test group leakage detected: {preview}")
    return len(train_groups), len(test_groups)
