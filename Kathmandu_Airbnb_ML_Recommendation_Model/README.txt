KATHMANDU AIRBNB ML ANALYSIS — MODEL 01

Files:
- accommodation_recommender.py: reusable recommender class and CLI.
- accommodation_recommender.joblib: fitted TF-IDF recommender artifact.
- properties_clean.csv: cleaned property input used to fit the model.
- sample_recommendations.csv: example output for Thamel, 2 guests, 1 bedroom,
  max NPR 5,000/night, and requested amenities.
- Kathmandu_Airbnb_Recommendation_Model.ipynb: walkthrough notebook.

Run:
  pip install -r requirements.txt
  python accommodation_recommender.py --neighbourhood Thamel --guests 2 --bedrooms 1 --max-budget 5000 --amenities Wi-Fi Kitchen "Air conditioning" --top-n 10

To fit and save:
  python accommodation_recommender.py --save-model --neighbourhood Thamel --guests 2

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
