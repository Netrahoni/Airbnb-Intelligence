KATHMANDU AIRBNB ML ANALYSIS — MODEL 01

Files:
- Kathmandu_Airbnb_Recommendation_Model.ipynb: walkthrough notebook.
- Shared recommendation implementation: src/advanced_recommender.py.
- Input dataset: data/processed/properties_clean.csv.
- Dashboard model artifact: models/property_recommender.joblib.

The notebook demonstrates personalized recommendations using the shared implementation and prepared dataset. Run it from the project environment.

To rebuild the dashboard recommendation model:
  python src/train_recommendation_model.py

Scoring:
- Hard filters: published, guest capacity, minimum bedrooms, maximum budget.
- Ranking: neighbourhood (0.20), property type (0.15), room type (0.10),
  TF-IDF amenity/description similarity (0.40), budget fit (0.15).
- Weights are renormalized over preferences supplied by the user.
- Rating and review_count are tie-breakers only.

Important limitations:
- Source property data identifies itself as Synthetic Sample Data.
- relevance_score is not a probability and not an evaluation metric.
- No verified user interaction data is available for collaborative filtering or
  offline ranking metrics.
