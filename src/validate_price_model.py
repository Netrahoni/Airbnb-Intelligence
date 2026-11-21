
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from src.model_evaluation import (
    regression_metrics,
    ensure_group_disjoint,
    validate_price_training_data,
)

# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/price_prediction_features.csv"
)

properties = pd.read_csv(
    "data/processed/properties_clean.csv"
)

df = df.merge(
    properties[["property_id", "nightly_price_npr"]],
    on="property_id",
    how="left",
    validate="many_to_one"
)

if df["nightly_price_npr"].isna().any():
    raise ValueError("Missing baseline property prices.")

print("Dataset:", df.shape)
print("Properties:", df["property_id"].nunique())

# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

categorical_features = [
    "city",
    "neighbourhood",
    "property_type",
    "room_type"
]

numerical_features = [
    "capacity",
    "bedrooms",
    "bathrooms",
    "lead_time_days",
    "stay_month",
    "stay_day",
    "stay_day_of_week",
    "nightly_price_npr"
]

text_features = ["amenities"]

feature_columns = (
    categorical_features
    + numerical_features
    + text_features
)

validate_price_training_data(
    df,
    feature_columns=feature_columns,
    target_column="base_price_npr",
    group_column="property_id",
)

X = df[feature_columns]
y = df["base_price_npr"]
groups = df["property_id"]

# --------------------------------------------------
# 3. Configure cross-validation
# --------------------------------------------------

group_kfold = GroupKFold(n_splits=5)

results = []

# --------------------------------------------------
# 4. Train and evaluate each fold
# --------------------------------------------------

for fold, (train_index, test_index) in enumerate(
    group_kfold.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    ensure_group_disjoint(groups.to_numpy(), train_index, test_index)
    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            ),
            (
                "amenities",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1
                ),
                "amenities"
            )
        ],
        sparse_threshold=0.0
    )

    # Gradient Boosting model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )

    print(f"\nTraining fold {fold}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Simple baseline prediction
    baseline_predictions = X_test["nightly_price_npr"]

    # Model metrics
    model_metrics = regression_metrics(y_test, predictions)
    model_mae = model_metrics["MAE"]
    model_rmse = model_metrics["RMSE"]
    model_r2 = model_metrics["R2"]

    # Baseline metrics
    baseline_metrics = regression_metrics(y_test, baseline_predictions)
    baseline_mae = baseline_metrics["MAE"]
    baseline_rmse = baseline_metrics["RMSE"]
    baseline_r2 = baseline_metrics["R2"]

    results.append({
        "fold": fold,
        "training_properties": len(train_groups),
        "testing_properties": len(test_groups),
        "GB_MAE": model_mae,
        "GB_RMSE": model_rmse,
        "GB_R2": model_r2,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_R2": baseline_r2
    })

    print("Gradient Boosting MAE:", round(model_mae, 2))
    print("Baseline MAE:", round(baseline_mae, 2))

# --------------------------------------------------
# 5. Save detailed results
# --------------------------------------------------

results_df = pd.DataFrame(results)

Path("reports").mkdir(exist_ok=True)

results_df.to_csv(
    "reports/price_model_cross_validation.csv",
    index=False
)

# --------------------------------------------------
# 6. Display average performance
# --------------------------------------------------

metric_columns = [
    "GB_MAE",
    "GB_RMSE",
    "GB_R2",
    "Baseline_MAE",
    "Baseline_RMSE",
    "Baseline_R2"
]

print("\n========== FIVE-FOLD RESULTS ==========")
print(results_df[["fold"] + metric_columns].round(4).to_string(index=False))

print("\n========== AVERAGE PERFORMANCE ==========")
print(results_df[metric_columns].mean().round(4))

print("\n========== PERFORMANCE VARIATION ==========")
print(results_df[metric_columns].std().round(4))

print("\nReport saved:")
print("reports/price_model_cross_validation.csv")

print("\nCross-validation completed successfully!")
