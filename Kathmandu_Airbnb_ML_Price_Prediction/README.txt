KATHMANDU AIRBNB ML ANALYSIS — MODEL 02: PRICE PREDICTION

Files:
- train_price_model.py: repeatable training and model-comparison script.
- price_prediction_model.joblib: selected fitted Ridge Regression pipeline.
- price_observations_clean.csv: training input copied from preprocessing output.
- model_comparison_metrics.csv: held-out metrics for five models.
- heldout_test_predictions.csv: predictions for unseen properties.
- training_metadata.json: split details, selected model, provenance, metrics.
- Kathmandu_Airbnb_Price_Prediction.ipynb: notebook walkthrough.

Run:
  pip install -r requirements.txt
  python train_price_model.py --data price_observations_clean.csv --output-dir model_output

Feature set:
Categorical: city, neighbourhood, property_type, room_type
Numeric: capacity, bedrooms, bathrooms, stay_month, stay_day_of_week, lead_days
Text: amenities
Target: base_price_npr

Evaluation:
A GroupShuffleSplit separates entire property IDs into train/test sets (80/20).
This avoids the same property appearing in both partitions.
Metrics: MAE, RMSE, R2.
Ridge selected for lowest RMSE and highest R2; Random Forest has lowest MAE.

Important:
The source data identifies itself as Synthetic Sample Data. Model metrics and predictions
are therefore experimental and must not be represented as verified real-world Kathmandu
Airbnb price performance. Prediction scores are not guaranteed market quotes.
