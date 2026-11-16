KATHMANDU AIRBNB ML ANALYSIS — PREPROCESSING OUTPUT

Contents:
- Kathmandu_Airbnb_ML_Data_Audit.ipynb: repeatable audit and preprocessing notebook.
- *_clean.csv: normalized working copies of non-empty datasets.
- *_schema_only.csv: original headers retained for empty datasets.
- dataset_quality_summary.csv: row/column/missing/duplicate overview.
- relationship_and_anomaly_checks.csv: ID joins and selected anomaly counts.

Important:
1. Original uploaded CSVs are not modified.
2. Properties and price observations identify their source as Synthetic Sample Data.
3. The hotel booking cancellation dataset is not established as Kathmandu Airbnb data.
4. Reviews and occupancy files are empty; no data has been fabricated.
5. Unusual booking records are flagged, not silently deleted.
6. Availability price values are retained, but a flag marks whether a property was available.
7. Missing URL columns that were entirely empty are removed from the cleaned property/price copies.
