
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advanced_recommender import AdvancedPropertyRecommender

DATA_PATH = ROOT / "data" / "processed" / "properties_clean.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== RECOMMENDATION WEIGHT OPTIMIZATION ==========\n")


# ============================================================
# 2. DEFINE WEIGHT CONFIGURATIONS
# ============================================================

WEIGHT_CONFIGS = {
    "Baseline": {
        "text_similarity": 0.30,
        "neighbourhood_match": 0.20,
        "budget_match": 0.20,
        "property_type_match": 0.10,
        "quality_score": 0.10,
        "capacity_match": 0.10,
    },

    "Balanced": {
        "text_similarity": 0.25,
        "neighbourhood_match": 0.20,
        "budget_match": 0.15,
        "property_type_match": 0.15,
        "quality_score": 0.10,
        "capacity_match": 0.15,
    },

    "Preference Focused": {
        "text_similarity": 0.20,
        "neighbourhood_match": 0.20,
        "budget_match": 0.15,
        "property_type_match": 0.20,
        "quality_score": 0.10,
        "capacity_match": 0.15,
    },

    "Text Focused": {
        "text_similarity": 0.35,
        "neighbourhood_match": 0.15,
        "budget_match": 0.15,
        "property_type_match": 0.10,
        "quality_score": 0.10,
        "capacity_match": 0.15,
    },

    "Location Focused": {
        "text_similarity": 0.20,
        "neighbourhood_match": 0.30,
        "budget_match": 0.15,
        "property_type_match": 0.15,
        "quality_score": 0.10,
        "capacity_match": 0.10,
    },
        "Preference-Location Hybrid": {
        "text_similarity": 0.20,
        "neighbourhood_match": 0.25,
        "budget_match": 0.15,
        "property_type_match": 0.15,
        "quality_score": 0.10,
        "capacity_match": 0.15,
    },

    "Preference-First Hybrid": {
        "text_similarity": 0.20,
        "neighbourhood_match": 0.22,
        "budget_match": 0.15,
        "property_type_match": 0.18,
        "quality_score": 0.10,
        "capacity_match": 0.15,
    },
}


# ============================================================
# 3. VALIDATE CONFIGURATIONS
# ============================================================

for name, weights in WEIGHT_CONFIGS.items():

    if set(weights) != set(AdvancedPropertyRecommender.WEIGHTS):
        raise ValueError(f"Invalid weight components in {name}")

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Weights for {name} must sum to 1.0.")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"Negative weight found in {name}")


# ============================================================
# 4. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

properties = pd.read_csv(DATA_PATH)

engine = AdvancedPropertyRecommender(properties)

published = engine.properties.copy()

print(f"Total dataset records: {len(properties)}")
print(f"Published properties: {len(published)}")

sample_size = min(25, len(published))

sample_properties = published.sample(
    n=sample_size,
    random_state=42,
).reset_index(drop=True)


# ============================================================
# 5. RUN WEIGHT EXPERIMENTS
# ============================================================

experiment_results = []
scenario_results = []

print("\n========== RUNNING EXPERIMENTS ==========\n")

original_weights = AdvancedPropertyRecommender.WEIGHTS.copy()

try:

    for config_name, weights in WEIGHT_CONFIGS.items():

        print(f"Evaluating: {config_name}")

        engine.WEIGHTS = weights.copy()

        for index, property_row in sample_properties.iterrows():

            base_price = float(property_row["nightly_price_npr"])

            if not np.isfinite(base_price) or base_price <= 0:
                continue

            guests = max(
                1,
                int(property_row["capacity"])
            )

            max_price = max(
                500,
                int(base_price * 1.5)
            )

            preferences = {
                "neighbourhood": property_row["neighbourhood"],
                "property_type": property_row["property_type"],
                "room_type": property_row["room_type"],
                "max_price": max_price,
                "guests": guests,
            }

            try:

                results = engine.recommend_personalized(
                    preferences,
                    number=10,
                )

                if results.empty:
                    continue

                prices = pd.to_numeric(
                    results["nightly_price_npr"],
                    errors="coerce",
                )

                capacities = pd.to_numeric(
                    results["capacity"],
                    errors="coerce",
                )

                # Price difference from reference property.
                price_difference = (
                    (prices - base_price).abs() / base_price * 100
                )

                scenario = {
                    "configuration": config_name,
                    "scenario_id": index + 1,
                    "result_count": len(results),

                    "average_recommendation_score": (
                        results["recommendation_score"].mean()
                    ),

                    "neighbourhood_match_pct": (
                        results["neighbourhood"]
                        .eq(property_row["neighbourhood"])
                        .mean() * 100
                    ),

                    "property_type_match_pct": (
                        results["property_type"]
                        .eq(property_row["property_type"])
                        .mean() * 100
                    ),

                    "room_type_match_pct": (
                        results["room_type"]
                        .eq(property_row["room_type"])
                        .mean() * 100
                    ),

                    "budget_compliance_pct": (
                        prices.dropna().le(max_price).mean() * 100
                        if prices.notna().any()
                        else np.nan
                    ),

                    "capacity_sufficiency_pct": (
                        capacities.dropna().ge(guests).mean() * 100
                        if capacities.notna().any()
                        else np.nan
                    ),

                    "average_price_difference_pct": (
                        price_difference.mean()
                    ),
                }

                scenario_results.append(scenario)

            except Exception as error:

                print(
                    f"[WARNING] {config_name}, "
                    f"scenario {index + 1}: {error}"
                )

finally:

    engine.WEIGHTS = original_weights


# ============================================================
# 6. GENERATE COMPARISON TABLE
# ============================================================

scenario_df = pd.DataFrame(scenario_results)

if scenario_df.empty:
    raise RuntimeError("No experiment results were generated.")

summary_df = (
    scenario_df
    .groupby("configuration")
    .agg(
        scenarios=("scenario_id", "nunique"),
        average_score=("average_recommendation_score", "mean"),
        neighbourhood_match=("neighbourhood_match_pct", "mean"),
        property_type_match=("property_type_match_pct", "mean"),
        room_type_match=("room_type_match_pct", "mean"),
        budget_compliance=("budget_compliance_pct", "mean"),
        capacity_sufficiency=("capacity_sufficiency_pct", "mean"),
        average_price_difference=("average_price_difference_pct", "mean"),
        average_result_count=("result_count", "mean"),
    )
    .reindex(WEIGHT_CONFIGS.keys())
    .round(2)
    .reset_index()
    .rename(columns={"index": "configuration"})
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n========== WEIGHT OPTIMIZATION RESULTS ==========\n")

print(summary_df.to_string(index=False))


# ============================================================
# 8. SAVE REPORTS
# ============================================================

summary_path = REPORT_DIR / "recommendation_weight_comparison.csv"

details_path = REPORT_DIR / "recommendation_weight_scenarios.csv"

weights_path = REPORT_DIR / "recommendation_weight_configurations.csv"

summary_df.to_csv(summary_path, index=False)

scenario_df.to_csv(details_path, index=False)

weights_df = pd.DataFrame(WEIGHT_CONFIGS).T
weights_df.index.name = "configuration"
weights_df.to_csv(weights_path)

print("\n========== REPORTS SAVED ==========\n")

print(summary_path)
print(details_path)
print(weights_path)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n========== EXPERIMENT COMPLETE ==========\n")

print(f"Configurations tested: {len(WEIGHT_CONFIGS)}")
print(f"Scenarios per configuration: {sample_size}")
print(f"Total successful evaluations: {len(scenario_df)}")

print(
    "\nNOTE: These results are heuristic comparisons "
    "using synthetic property data and proxy relevance measures."
)

print(
    "Do not select a configuration using average recommendation "
    "score alone. Consider preference matches, price differences, "
    "budget compliance, capacity and result availability."
)
