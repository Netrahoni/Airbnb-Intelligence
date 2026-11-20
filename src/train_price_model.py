
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "price_prediction_features.csv"
PROPERTIES_PATH = ROOT / "data" / "processed" / "properties_clean.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


print("\n========== AIRBNB INTELLIGENCE | PRICE PREDICTION ==========")


# ============================================================
# 2. LOAD DATASETS
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not PROPERTIES_PATH.exists():
    raise FileNotFoundError(f"Properties dataset not found: {PROPERTIES_PATH}")

df = pd.read_csv(DATA_PATH)

properties = pd.read_csv(PROPERTIES_PATH)

print("\nPrice dataset shape:", df.shape)
print("Properties dataset shape:", properties.shape)


# ============================================================
# 3. ADD BASELINE NIGHTLY PRICE FEATURE
# ============================================================

required_property_columns = [
    "property_id",
    "nightly_price_npr"
]

missing_columns = [
    column for column in required_property_columns
    if column not in properties.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in properties dataset: {missing_columns}"
    )

df = df.merge(
    properties[required_property_columns],
    on="property_id",
    how="left",
    validate="many_to_one"
)

if df["nightly_price_npr"].isna().any():
    raise ValueError("Missing baseline property prices after merging.")

print("\nBaseline nightly price feature added successfully.")


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

categorical_features = [
    "city",
    "neighbourhood",
    "property_type",
    "room_type"
]

numeric_features = [
    "capacity",
    "bedrooms",
    "bathrooms",
    "lead_time_days",
    "stay_month",
    "stay_day",
    "stay_day_of_week",
    "nightly_price_npr"
]

feature_columns = (
    categorical_features
    + numeric_features
    + ["amenities"]
)

target_column = "base_price_npr"

required_columns = feature_columns + [
    target_column,
    "property_id"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

if df[required_columns].isna().any().any():
    raise ValueError("Missing values detected in required model columns.")

X = df[feature_columns].copy()
y = df[target_column].copy()

print("\nFeatures:", len(feature_columns))
print("Target:", target_column)


# ============================================================
# 5. SPLIT DATA BY PROPERTY ID
# ============================================================

# Group-based splitting ensures that observations belonging
# to the same property do not appear in both training and testing.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(
        X,
        y,
        groups=df["property_id"]
    )
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

train_properties = set(
    df.iloc[train_index]["property_id"]
)

test_properties = set(
    df.iloc[test_index]["property_id"]
)

overlapping_properties = train_properties & test_properties

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training properties:", len(train_properties))
print("Testing properties:", len(test_properties))
print("Overlapping property IDs:", len(overlapping_properties))

if overlapping_properties:
    raise ValueError("Data leakage detected: property IDs overlap.")


# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numeric_features
        ),
        (
            "amenities",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1
            ),
            "amenities"
        )
    ],
    sparse_threshold=0.0
)


# ============================================================
# 7. CREATE GRADIENT BOOSTING MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n========== MODEL TRAINING ==========")
print("Training Gradient Boosting Regressor...")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ============================================================
# 9. PREDICT PRICES
# ============================================================

predictions = model.predict(X_test)

print("\nPrice predictions generated successfully.")


# ============================================================
# 10. EVALUATE MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== MODEL EVALUATION ==========")

print(f"Mean Absolute Error (MAE): NPR {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): NPR {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")


# ============================================================
# 11. COMPARE WITH BASELINE PREDICTION
# ============================================================

# Baseline prediction uses the property's existing listed price.
# This is a useful comparison to determine whether ML adds value.

baseline_predictions = df.iloc[test_index][
    "nightly_price_npr"
].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

print("\n========== BASELINE COMPARISON ==========")

print(f"Baseline MAE: NPR {baseline_mae:.2f}")
print(f"Baseline RMSE: NPR {baseline_rmse:.2f}")
print(f"Baseline R2 Score: {baseline_r2:.4f}")

improvement = (
    (baseline_mae - mae) / baseline_mae
) * 100

print(f"MAE improvement over baseline: {improvement:.2f}%")


# ============================================================
# 12. SAVE TRAINED MODEL
# ============================================================

model_path = MODEL_DIR / "price_predictor.joblib"

joblib.dump(model, model_path)

print("\nTrained model saved:", model_path)


# ============================================================
# 13. SAVE MODEL EVALUATION REPORT
# ============================================================

metrics = pd.DataFrame([
    {"metric": "MAE", "value": mae},
    {"metric": "RMSE", "value": rmse},
    {"metric": "R2", "value": r2},
    {"metric": "Baseline MAE", "value": baseline_mae},
    {"metric": "Baseline RMSE", "value": baseline_rmse},
    {"metric": "Baseline R2", "value": baseline_r2},
    {"metric": "MAE Improvement Percentage", "value": improvement},
    {"metric": "Training rows", "value": len(X_train)},
    {"metric": "Testing rows", "value": len(X_test)},
    {"metric": "Training properties", "value": len(train_properties)},
    {"metric": "Testing properties", "value": len(test_properties)},
    {"metric": "Overlapping properties", "value": len(overlapping_properties)}
])

metrics_path = REPORT_DIR / "price_model_metrics.csv"

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 14. SAVE ACTUAL VS PREDICTED PRICE REPORT
# ============================================================

test_records = df.iloc[test_index].copy()

prediction_results = pd.DataFrame({
    "property_id": test_records["property_id"].values,
    "city": test_records["city"].values,
    "neighbourhood": test_records["neighbourhood"].values,
    "property_type": test_records["property_type"].values,
    "room_type": test_records["room_type"].values,
    "capacity": test_records["capacity"].values,
    "lead_time_days": test_records["lead_time_days"].values,
    "nightly_price_npr": test_records["nightly_price_npr"].values,
    "actual_price_npr": y_test.values,
    "predicted_price_npr": predictions,
    "baseline_price_prediction_npr": baseline_predictions,
    "absolute_error_npr": np.abs(
        y_test.values - predictions
    ),
    "signed_error_npr": predictions - y_test.values
})

prediction_results["percentage_error"] = (
    prediction_results["absolute_error_npr"]
    / prediction_results["actual_price_npr"].replace(0, np.nan)
) * 100

prediction_path = REPORT_DIR / "price_predictions.csv"

prediction_results.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 15. DISPLAY FINAL RESULTS
# ============================================================

print("\n========== REPORTS SAVED ==========")

print("Model:", model_path)
print("Evaluation report:", metrics_path)
print("Prediction report:", prediction_path)

print("\n========== SAMPLE PRICE PREDICTIONS ==========")

print(
    prediction_results.head(10).round(2).to_string(index=False)
)

print("\n========== TRAINING COMPLETED SUCCESSFULLY ==========")
