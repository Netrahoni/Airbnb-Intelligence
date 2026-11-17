
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = Path("data/processed/price_prediction_features.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Load property-level baseline prices
properties = pd.read_csv(
    "data/processed/properties_clean.csv"
)

# Add baseline nightly price using property ID
df = df.merge(
    properties[["property_id", "nightly_price_npr"]],
    on="property_id",
    how="left",
    validate="many_to_one"
)

if df["nightly_price_npr"].isna().any():
    raise ValueError("Some properties have missing baseline prices.")

print("Baseline price feature added successfully.")

print("\n========== PRICE MODEL COMPARISON ==========")
print("Dataset shape:", df.shape)

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

X = df[feature_columns].copy()
y = df["base_price_npr"].copy()
groups = df["property_id"].copy()

# --------------------------------------------------
# 3. Split by property ID
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

train_properties = set(groups.iloc[train_index])
test_properties = set(groups.iloc[test_index])

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training properties:", len(train_properties))
print("Testing properties:", len(test_properties))
print("Overlapping properties:", len(
    train_properties.intersection(test_properties)
))

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

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
# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

results = []

for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

    print("MAE: NPR", round(mae, 2))
    print("RMSE: NPR", round(rmse, 2))
    print("R2:", round(r2, 4))

# --------------------------------------------------
# 7. Compare results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 8. Save comparison report
# --------------------------------------------------

output_path = REPORT_DIR / "price_model_comparison.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\nComparison report saved:", output_path)
print("\nModel comparison completed successfully!")
