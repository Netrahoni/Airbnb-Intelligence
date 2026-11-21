AIRBNB INTELLIGENCE — MODEL 01: ACCOMMODATION RECOMMENDATION

Files:
- Kathmandu_Airbnb_Recommendation_Model.ipynb: walkthrough notebook.
- Shared recommendation implementation: src/advanced_recommender.py.
- Input dataset: data/processed/properties_clean.csv.
- Dashboard model artifact: models/property_recommender.joblib.

The notebook demonstrates personalized recommendations using the shared
implementation and prepared dataset. Run it from the project environment.

To rebuild the dashboard recommendation model:
  python src/train_recommendation_model.py

Current default ranking weights (src/advanced_recommender.py):
- Text similarity (TF-IDF over amenities and description): 0.20
- Neighbourhood match: 0.22
- Property-type/room-type match: 0.18 combined
- Quality score (rating and review-count signal): 0.10
- Guest capacity match: 0.15
- Budget compliance: 0.15

Personalization behavior:
- Explicit neighbourhood, property/room type, text, and guest preferences
  activate their corresponding ranking signals; active weights are
  renormalized while retaining the quality signal.
- Property type and room type share one combined matching component.
- Maximum budget is an affordability ceiling and hard filter, not a target
  spend or a reward for spending closer to the limit.
- Guest capacity, minimum bedrooms, and maximum budget are hard constraints
  where supplied.
- Ratings and review counts contribute to the quality signal and tie-breaking.

Evaluation limitations:
- Scenario-based scores are heuristic preference-match diagnostics, not
  ground-truth precision, recall, NDCG, or user satisfaction.
- Source property data identifies itself as Synthetic Sample Data.
- No verified user interaction, click, or booking labels are available.
- relevance_score is a ranking score, not a probability.
- Do not present the current output as verified marketplace recommendations.
