from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "processed"

print("\n========== FEATURE ENGINEERING ==========")

# Load datasets
properties = pd.read_csv(DATA / "properties_clean.csv")
prices = pd.read_csv(DATA / "price_observations_clean.csv")

# ---------------------------------------
# 1. PROPERTY RECOMMENDATION DATASET
# ---------------------------------------

recommendation_columns = [
    "property_id",
    "property_name",
    "city",
    "neighbourhood",
    "property_type",
    "room_type",
    "capacity",
    "bedrooms",
    "bathrooms",
    "amenities",
    "nightly_price_npr",
    "rating",
    "review_count",
    "description",
    "verified",
    "published"
]

recommendation = properties[recommendation_columns].copy()

recommendation.to_csv(
    OUTPUT / "recommendation_features.csv",
    index=False
)

print("\nRecommendation dataset created.")
print("Rows:", len(recommendation))
print("Columns:", len(recommendation.columns))

# ---------------------------------------
# 2. PRICE PREDICTION DATASET
# ---------------------------------------

# The price dataset already contains property details.
# Therefore, we do not need another merge.

price_data = prices.copy()

# Convert date columns
price_data["observation_date"] = pd.to_datetime(
    price_data["observation_date"],
    errors="coerce"
)

price_data["stay_date"] = pd.to_datetime(
    price_data["stay_date"],
    errors="coerce"
)

# Calculate days between observation and stay
price_data["lead_time_days"] = (
    price_data["stay_date"] - price_data["observation_date"]
).dt.days

# Extract date-based features
price_data["stay_month"] = price_data["stay_date"].dt.month
price_data["stay_day"] = price_data["stay_date"].dt.day

# Pandas 4 compatible property
price_data["stay_day_of_week"] = (
    price_data["stay_date"].dt.day_of_week
)

# Select model features
price_model_columns = [
    "property_id",
    "city",
    "neighbourhood",
    "property_type",
    "room_type",
    "capacity",
    "bedrooms",
    "bathrooms",
    "amenities",
    "lead_time_days",
    "stay_month",
    "stay_day",
    "stay_day_of_week",
    "base_price_npr"
]

price_model = price_data[price_model_columns].copy()

# Remove rows with missing target or lead time
price_model = price_model.dropna(
    subset=["base_price_npr", "lead_time_days"]
)

# Save prepared dataset
price_model.to_csv(
    OUTPUT / "price_prediction_features.csv",
    index=False
)

print("\nPrice prediction dataset created.")
print("Rows:", len(price_model))
print("Columns:", len(price_model.columns))

# ---------------------------------------
# 3. SUMMARY
# ---------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("\nRecommendation dataset:")
print(recommendation.head(3).to_string(index=False))

print("\nPrice prediction dataset:")
print(price_model.head(3).to_string(index=False))

print("\nPrice target statistics:")
print(price_model["base_price_npr"].describe())

print("\nSaved files:")
print(OUTPUT / "recommendation_features.csv")
print(OUTPUT / "price_prediction_features.csv")

print("\nFeature engineering completed successfully!")
