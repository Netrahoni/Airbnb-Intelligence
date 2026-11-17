"""
Model 04 - Booking Cancellation Prediction: preprocessing (Step 2)

What this file does
-------------------
1. Loads data/processed/booking_cancellations_clean.csv
2. Removes logically invalid rows (negative ADR, zero guests)
3. Builds model features and drops columns that leak or duplicate information
4. Defines the numeric and categorical preprocessing pipelines
5. Makes a stratified 80/20 train/test split (plus an optional time-based split)

IMPORTANT: the data is general hotel booking data, NOT verified Kathmandu
Airbnb data. Results describe a method, not the Kathmandu market.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Project root = the folder that contains src/, data/, reports/, models/
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "booking_cancellations_clean.csv"

RANDOM_STATE = 42
TARGET = "is_canceled"

# --- Columns removed because they identify, duplicate or are derived -------
# booking_id        : identifier only
# booking_date      : equals arrival_date - lead_time in 100% of rows (verified)
# arrival_date      : replaced by arrival_month / arrival_weekday
# total_nights      : equals weekend nights + week nights (verified)
# *_flag columns    : audit flags built from other columns
DROP_COLUMNS = [
    "booking_id", "booking_date", "arrival_date", "total_nights",
    "adr_is_negative", "guest_count_is_zero", "adr_iqr_outlier_flag",
]

# --- Feature groups ---------------------------------------------------------
# Skewed numbers: log1p squeezes very large values (e.g. lead_time up to 737)
LOG_NUMERIC = ["lead_time", "adr", "previous_cancellations",
               "previous_bookings_not_canceled"]
# Other numbers: only scaled
PLAIN_NUMERIC = ["stays_in_weekend_nights", "stays_in_week_nights", "adults",
                 "children", "babies", "total_guests",
                 "total_of_special_requests", "is_repeated_guest"]
CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel",
               "deposit_type", "customer_type", "arrival_month",
               "arrival_weekday"]


def load_booking_data(path=DATA_PATH):
    """Read the cleaned CSV."""
    return pd.read_csv(path)


def remove_invalid_rows(df):
    """Drop rows that cannot be real bookings. Returns (clean_df, counts)."""
    negative_adr = df["adr"] < 0
    zero_guests = (df["adults"] + df["children"] + df["babies"]) == 0
    counts = {
        "negative_adr_rows": int(negative_adr.sum()),
        "zero_guest_rows": int(zero_guests.sum()),
        "removed_total": int((negative_adr | zero_guests).sum()),
    }
    return df.loc[~(negative_adr | zero_guests)].copy(), counts


def build_features(df, use_deposit_type=True):
    """Create X (features) and y (target) from the cleaned bookings."""
    df = df.copy()
    arrival = pd.to_datetime(df["arrival_date"], utc=True)
    df["arrival_month"] = arrival.dt.month.astype(str)       # '1'..'12'
    df["arrival_weekday"] = arrival.dt.day_name()
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_repeated_guest"] = df["is_repeated_guest"].astype(int)

    y = df[TARGET].astype(int)
    X = df.drop(columns=DROP_COLUMNS + [TARGET])
    if not use_deposit_type:
        X = X.drop(columns=["deposit_type"])
    return X, y


def make_preprocessor(use_deposit_type=True):
    """Build the ColumnTransformer (fit it on TRAINING data only)."""
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cats = [c for c in CATEGORICAL if use_deposit_type or c != "deposit_type"]
    return ColumnTransformer([
        ("log_num", log_pipe, LOG_NUMERIC),
        ("num", num_pipe, PLAIN_NUMERIC),
        ("cat", cat_pipe, cats),
    ])


def stratified_split(X, y, test_size=0.20):
    """Random 80/20 split that keeps the same cancellation rate in both parts."""
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=RANDOM_STATE)


def time_based_split(df, test_fraction=0.20):
    """Stricter check: train on earlier arrivals, test on the latest ones."""
    order = pd.to_datetime(df["arrival_date"], utc=True).sort_values().index
    cut = int(len(order) * (1 - test_fraction))
    return df.loc[order[:cut]], df.loc[order[cut:]]
