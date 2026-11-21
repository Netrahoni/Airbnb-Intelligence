"""Reusable, side-effect-free data quality profiling helpers."""

import numpy as np
import pandas as pd


def profile_dataframe(
    frame,
    dataset_name,
    key_columns=(),
    numeric_bounds=None,
):
    """Return one quality-profile row per column, including dataset-level checks."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    row_count = len(frame)
    missing_keys = sorted(set(key_columns).difference(frame.columns))
    if missing_keys:
        raise ValueError(f"{dataset_name} is missing key columns: {missing_keys}")

    duplicate_key_rows = (
        int(frame.duplicated(subset=list(key_columns)).sum())
        if key_columns and row_count else 0
    )
    bounds = numeric_bounds or {}
    rows = []

    for column in frame.columns:
        values = frame[column]
        missing_count = int(values.isna().sum())
        invalid_numeric_count = 0
        out_of_range_count = 0

        if column in bounds:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid_numeric_count = int((values.notna() & numeric.isna()).sum())
            lower, upper = bounds[column]
            valid = numeric.notna()
            if lower is not None:
                valid &= numeric >= lower
            if upper is not None:
                valid &= numeric <= upper
            out_of_range_count = int((numeric.notna() & ~valid).sum())

        rows.append({
            "dataset": str(dataset_name),
            "column": str(column),
            "row_count": row_count,
            "missing_count": missing_count,
            "missing_percent": round(missing_count / row_count * 100, 2) if row_count else 0.0,
            "duplicate_key_rows": duplicate_key_rows,
            "invalid_numeric_count": invalid_numeric_count,
            "out_of_range_count": out_of_range_count,
        })

    return pd.DataFrame(rows, columns=[
        "dataset", "column", "row_count", "missing_count", "missing_percent",
        "duplicate_key_rows", "invalid_numeric_count", "out_of_range_count",
    ])


def build_data_quality_report(datasets):
    """Combine named DataFrames into a single quality report."""
    reports = [
        profile_dataframe(
            frame,
            name,
            key_columns=options.get("key_columns", ()),
            numeric_bounds=options.get("numeric_bounds"),
        )
        for name, frame, options in datasets
    ]
    if not reports:
        return pd.DataFrame(columns=[
            "dataset", "column", "row_count", "missing_count", "missing_percent",
            "duplicate_key_rows", "invalid_numeric_count", "out_of_range_count",
        ])
    return pd.concat(reports, ignore_index=True)
