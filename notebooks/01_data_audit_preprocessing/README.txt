KATHMANDU AIRBNB ML ANALYSIS — PREPROCESSING OUTPUT

Contents:
- Kathmandu_Airbnb_ML_Data_Audit.ipynb: repeatable audit and preprocessing notebook.
- Output files are written to data/audit_output/ from the repository root.

Important:
1. Put the seven original source CSVs in data/raw/ using the filenames listed in the notebook before running it. Raw source files are not included in this repository.
2. The notebook writes audit outputs to data/audit_output/ and does not overwrite the dashboard's prepared data in data/processed/.
3. Properties and price observations identify their source as Synthetic Sample Data.
4. The hotel booking cancellation dataset is not established as Kathmandu Airbnb data.
5. Reviews and occupancy files are empty; no data has been fabricated.
6. Unusual booking records are flagged, not silently deleted.
7. Availability price values are retained, but a flag marks whether a property was available.
8. Missing URL columns that were entirely empty are removed from the cleaned property/price copies.
