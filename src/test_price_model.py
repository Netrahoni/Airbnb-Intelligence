
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "price_predictor.joblib"

PRICE_DATA_PATH = (
    ROOT / "data" / "processed" / "price_prediction_features.csv"
)

PROPERTIES_PATH = (
    ROOT / "data" / "processed" / "properties_clean.csv"
)


print("\n========== SAVED PRICE MODEL TEST ==========")


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

for path in [MODEL_PATH, PRICE_DATA_PATH, PROPERTIES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# ============================================================
# 3. LOAD SAVED MODEL
# ============================================================

print("\nLoading saved model...")

model = joblib.load(MODEL_PATH)

print("Saved model loaded successfully!")


# ============================================================
# 4. LOAD DATASETS
# ============================================================

price_data = pd.read_csv(PRICE_DATA_PATH)

properties = pd.read_csv(PROPERTIES_PATH)

print("\nPrice dataset:", price_data.shape)
print("Properties dataset:", properties.shape)


# ============================================================
# 5. ADD BASELINE PRICE FEATURE
# ============================================================

price_data = price_data.merge(
    properties[["property_id", "nightly_price_npr"]],
    on="property_id",
    how="left",
    validate="many_to_one"
)

if price_data["nightly_price_npr"].isna().any():
    raise ValueError("Missing baseline property prices.")


# ============================================================
# 6. PREPARE MODEL INPUT FEATURES
# ============================================================

feature_columns = [
    "city",
    "neighbourhood",
    "property_type",
    "room_type",
    "capacity",
    "bedrooms",
    "bathrooms",
    "lead_time_days",
    "stay_month",
    "stay_day",
    "stay_day_of_week",
    "nightly_price_npr",
    "amenities"
]

missing_columns = [
    column for column in feature_columns
    if column not in price_data.columns
]

if missing_columns:
    raise ValueError(f"Missing model features: {missing_columns}")


# Select one observation
sample = price_data.iloc[[0]].copy()

X_sample = sample[feature_columns].copy()


# ============================================================
# 7. GENERATE PREDICTION
# ============================================================

print("\nGenerating price prediction...")

predicted_price = model.predict(X_sample)[0]

actual_price = sample["base_price_npr"].iloc[0]

baseline_price = sample["nightly_price_npr"].iloc[0]

if not np.isfinite(predicted_price):
    raise ValueError("Model returned an invalid prediction.")


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========== PREDICTION RESULTS ==========")

print("Property ID:", sample["property_id"].iloc[0])
print("City:", sample["city"].iloc[0])
print("Neighbourhood:", sample["neighbourhood"].iloc[0])
print("Property type:", sample["property_type"].iloc[0])
print("Room type:", sample["room_type"].iloc[0])

print("\nBaseline price: NPR", round(baseline_price, 2))
print("Actual observed price: NPR", round(actual_price, 2))
print("Predicted price: NPR", round(predicted_price, 2))

absolute_error = abs(actual_price - predicted_price)

print("Absolute prediction error: NPR", round(absolute_error, 2))


# ============================================================
# 9. FINAL TEST STATUS
# ============================================================

print("\n========== TEST STATUS ==========")

print("SUCCESS: Model loaded and prediction completed.")
print("Input feature count:", len(feature_columns))
print("Prediction is finite:", np.isfinite(predicted_price))

print("\nSaved model testing completed successfully!")
