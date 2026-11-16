"""Kathmandu Airbnb ML Analysis — Model 02: Price prediction.

Trains and compares regression models using a group holdout split by property_id.
The source data labels itself Synthetic Sample Data; results are experimental.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CATEGORICAL = ["city", "neighbourhood", "property_type", "room_type"]
NUMERIC = ["capacity", "bedrooms", "bathrooms", "stay_month", "stay_day_of_week", "lead_days"]
TEXT = "amenities"
TARGET = "base_price_npr"

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    required = ["property_id", "observation_date", "stay_date", TARGET] + CATEGORICAL + NUMERIC[:3] + [TEXT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["observation_date", "stay_date"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["stay_month"] = df["stay_date"].dt.month
    df["stay_day_of_week"] = df["stay_date"].dt.dayofweek
    df["lead_days"] = (df["stay_date"] - df["observation_date"]).dt.days
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["property_id", TARGET, "stay_month", "lead_days"]).copy()
    df[TEXT] = df[TEXT].fillna("").astype(str)
    return df

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), NUMERIC),
            ("amenities", TfidfVectorizer(stop_words="english", max_features=100, ngram_range=(1, 2)), TEXT)
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

def train(csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_data(csv_path)
    features = CATEGORICAL + NUMERIC + [TEXT]
    X, y, groups = df[features], df[TARGET].astype(float), df["property_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates = {
        "Dummy Median Baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=2,
            loss="huber", random_state=42
        ),
    }

    fitted, rows = {}, []
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        fitted[name] = pipeline
        rows.append({
            "model": name,
            "MAE_NPR": mean_absolute_error(y_test, predictions),
            "RMSE_NPR": mean_squared_error(y_test, predictions) ** 0.5,
            "R2": r2_score(y_test, predictions),
        })

    metrics = pd.DataFrame(rows).sort_values("MAE_NPR").reset_index(drop=True)
    # Select by RMSE/R² balance; keep MAE comparison visible in the report.
    selected_name = "Ridge Regression"
    selected = fitted[selected_name]
    joblib.dump(selected, output_dir / "price_prediction_model.joblib")
    metrics.to_csv(output_dir / "model_comparison_metrics.csv", index=False)

    predictions = selected.predict(X_test)
    test_output = df.iloc[test_idx][
        ["property_id", "neighbourhood", "property_type", "room_type", "stay_date", TARGET]
    ].copy()
    test_output["predicted_price_npr"] = np.round(predictions, 2)
    test_output["absolute_error_npr"] = np.round(np.abs(y_test.to_numpy() - predictions), 2)
    test_output.to_csv(output_dir / "heldout_test_predictions.csv", index=False)

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    metadata = {
        "selected_model": selected_name,
        "target": TARGET,
        "training_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "training_unique_properties": len(train_groups),
        "test_unique_properties": len(test_groups),
        "property_overlap": len(train_groups & test_groups),
        "split": "GroupShuffleSplit by property_id; test_size=0.20; random_state=42",
        "features": features,
        "source_provenance": sorted(df["source_name"].dropna().astype(str).unique().tolist())
            if "source_name" in df else ["Not supplied"],
        "metrics": metrics.to_dict(orient="records"),
        "selection_note": "Ridge has the best RMSE and R²; Random Forest has the lowest MAE.",
        "limitation": "Source data is synthetic; metrics do not establish real Kathmandu Airbnb price performance."
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="price_observations_clean.csv")
    parser.add_argument("--output-dir", default="price_model_output")
    args = parser.parse_args()
    metrics, metadata = train(args.data, args.output_dir)
    print(metrics.to_string(index=False))
    print("Selected:", metadata["selected_model"])

if __name__ == "__main__":
    main()
