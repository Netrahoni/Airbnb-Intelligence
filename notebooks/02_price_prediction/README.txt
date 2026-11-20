KATHMANDU AIRBNB ML ANALYSIS — MODEL 02: PRICE PREDICTION

Files:
- Kathmandu_Airbnb_Price_Prediction.ipynb: notebook walkthrough.
- Training code: src/train_price_model.py.
- Input datasets: data/processed/price_prediction_features.csv and data/processed/properties_clean.csv.
- Trained model and reports: models/price_predictor.joblib and reports/.

Run:
  python src/train_price_model.py

Feature set:
Categorical: city, neighbourhood, property_type, room_type
Numeric: capacity, bedrooms, bathrooms, stay_month, stay_day_of_week, lead_days
Text: amenities
Target: base_price_npr

Evaluation:
A GroupShuffleSplit separates entire property IDs into train/test sets.
This avoids the same property appearing in both partitions.
Metrics include MAE, RMSE, and R2, compared with a median-price baseline.

Important:
The source data identifies itself as Synthetic Sample Data. Model metrics and predictions are experimental and must not be represented as verified real-world Kathmandu Airbnb price performance. Prediction scores are not guaranteed market quotes.
