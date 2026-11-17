
import joblib
import pandas as pd
from pathlib import Path

print("========== TESTING SAVED PRICE MODEL ==========")

# Load the saved model
model_path = Path("models/price_predictor.joblib")
model = joblib.load(model_path)

print("Model loaded successfully!")

# Load the prepared dataset
df = pd.read_csv("data/processed/price_prediction_features.csv")

# Select the same input features used during training
features = [
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
    "amenities"
]

# Select one sample for prediction
sample = df[features].iloc[[0]]

# Predict the nightly price
predicted_price = model.predict(sample)[0]

print("\n========== SAMPLE PRICE PREDICTION ==========")
print("Property:", df.iloc[0]["property_id"])
print("Actual price: NPR", df.iloc[0]["base_price_npr"])
print("Predicted price: NPR", round(predicted_price, 2))

print("\nSaved model test completed successfully!")
