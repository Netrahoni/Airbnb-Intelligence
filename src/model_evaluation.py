"""Pure evaluation helpers used by model workflows and automated tests."""

import pandas as pd

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.size == 0:
        raise ValueError("Evaluation data must contain at least one observation.")
    if actual.size < 2:
        raise ValueError("R2 evaluation requires at least two observations.")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have the same length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation values must all be finite.")
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }



def price_band_diagnostics(predictions):
    """Summarise held-out prediction errors by actual-price band."""
    required = {"actual_price_npr", "predicted_price_npr"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction report is missing columns: {missing}")

    frame = predictions[list(required)].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    columns = [
        "price_category", "sample_count", "average_actual_price",
        "average_predicted_price", "mean_absolute_error",
        "mean_percentage_error", "average_signed_error",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    actual = frame["actual_price_npr"].to_numpy(dtype=float)
    predicted = frame["predicted_price_npr"].to_numpy(dtype=float)
    absolute_error = np.abs(predicted - actual)
    frame["absolute_error"] = absolute_error
    frame["percentage_error"] = np.divide(
        absolute_error * 100, np.abs(actual),
        out=np.full(len(actual), np.nan, dtype=float), where=actual != 0,
    )
    frame["signed_error"] = predicted - actual
    frame["price_category"] = pd.cut(
        frame["actual_price_npr"],
        bins=[-np.inf, 2000, 4000, 6000, np.inf],
        labels=[
            "Budget (up to 2000)", "Mid-range (2001-4000)",
            "Premium (4001-6000)", "Luxury (above 6000)",
        ],
        include_lowest=True,
    )
    return (
        frame.groupby("price_category", observed=True, sort=True)
        .agg(
            sample_count=("absolute_error", "size"),
            average_actual_price=("actual_price_npr", "mean"),
            average_predicted_price=("predicted_price_npr", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_percentage_error=("percentage_error", "mean"),
            average_signed_error=("signed_error", "mean"),
        )
        .reset_index()
    )

def ensure_group_disjoint(groups, train_indices, test_indices):
    groups = np.asarray(groups)
    train_groups = set(groups[np.asarray(train_indices, dtype=int)])
    test_groups = set(groups[np.asarray(test_indices, dtype=int)])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        preview = ", ".join(sorted(map(str, overlap))[:5])
        raise ValueError(f"Train/test group leakage detected: {preview}")
    return len(train_groups), len(test_groups)



def validate_price_training_data(
    frame,
    feature_columns,
    target_column="base_price_npr",
    group_column="property_id",
    minimum_groups=5,
):
    """Fail early on unusable targets, features, or insufficient property groups."""
    required = list(dict.fromkeys(
        list(feature_columns) + [target_column, group_column]
    ))
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Price training data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Price training data must contain at least one row.")
    if frame[required].isna().any().any():
        raise ValueError("Price training data contains missing required values.")

    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Price target values must all be finite numbers.")
    if (target <= 0).any():
        raise ValueError("Price target values must all be greater than zero.")

    if "lead_time_days" in frame.columns:
        lead_time = pd.to_numeric(frame["lead_time_days"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(lead_time).all() or (lead_time < 0).any():
            raise ValueError("Lead time must contain finite, non-negative values.")

    group_count = frame[group_column].nunique(dropna=True)
    if group_count < minimum_groups:
        raise ValueError(
            f"Price training requires at least {minimum_groups} distinct "
            f"{group_column} groups; found {group_count}."
        )
    return int(group_count)
