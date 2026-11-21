AIRBNB INTELLIGENCE — MODEL 02: PRICE PREDICTION

Purpose
Demonstrate a reproducible regression workflow for estimating the prepared
synthetic observation target, base_price_npr, using property attributes,
amenities, stay timing, and the property's existing nightly listing price.

Current training implementation
- Script: src/train_price_model.py
- Model: GradientBoostingRegressor (150 estimators, learning_rate=0.05,
  max_depth=3, random_state=42)
- Target: base_price_npr
- Categorical: city, neighbourhood, property_type, room_type
- Numeric: capacity, bedrooms, bathrooms, lead_time_days, stay_month,
  stay_day, stay_day_of_week, nightly_price_npr
- Text: amenities
- Split: GroupShuffleSplit by property_id, test_size=0.20, random_state=42
- Training: 808 observations / 320 properties
- Test: 192 observations / 80 properties
- Property overlap: 0

Baseline and interpretation
The baseline prediction is the property's existing nightly_price_npr, merged
from properties_clean.csv by property_id. It is NOT a median-only baseline.
The model therefore estimates synthetic observed base_price_npr conditional
on an already-known listing price and stay features. It is not an independent
market-price estimator for a listing with no known price.

Latest saved evaluation (reports/price_model_metrics.csv)
- Gradient Boosting MAE: NPR 148.33
- Gradient Boosting RMSE: NPR 218.93
- Gradient Boosting R2: 0.9942
- Existing-listing-price baseline MAE: NPR 158.85
- Existing-listing-price baseline RMSE: NPR 225.64
- Existing-listing-price baseline R2: 0.9938
- MAE reduction vs baseline: 6.62%

Data provenance and limitations
The prepared price observations identify their source as Synthetic Sample Data.
The 1,000 observations from 400 property IDs are suitable for workflow
demonstration and testing only. These results do not establish real Kathmandu
Airbnb price accuracy, verified market performance, or deployable pricing
advice. A later-time evaluation using permitted, timestamped, independently
verified market observations is required before making real-world claims.

Run from the repository root:
  python src/train_price_model.py
  python src/validate_price_model.py

Generated artifacts:
- models/price_predictor.joblib
- reports/price_model_metrics.csv
- reports/price_predictions.csv
- reports/price_model_cross_validation.csv
