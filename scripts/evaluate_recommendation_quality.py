
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

print("\n========== RECOMMENDATION QUALITY EVALUATION ==========\n")


# ============================================================
# 2. LOAD DATA AND INITIALIZE ENGINE
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

properties = pd.read_csv(DATA_PATH)

engine = AdvancedPropertyRecommender(properties)

published = engine.properties.copy()

print(f"Dataset records: {len(properties)}")
print(f"Published properties: {len(published)}")

if published.empty:
    raise ValueError("No published properties available for evaluation.")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_mean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    return round(values.mean(), 2) if values.notna().any() else np.nan


def safe_number(value, default=0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return default


def get_score(results):
    if results.empty or "recommendation_score" not in results.columns:
        return np.nan

    return safe_mean(results["recommendation_score"])


def add_result(rows, result):
    rows.append(result)


# ============================================================
# 4. PREPARE EVALUATION SCENARIOS
# ============================================================

# Use a reproducible sample of properties as recommendation
# reference points and preference profiles.

sample_size = min(25, len(published))

sample_properties = published.sample(
    n=sample_size,
    random_state=42
).reset_index(drop=True)

scenario_results = []
similarity_results = []
component_results = []
execution_results = []

print("\n========== PERSONALIZED RECOMMENDATION QUALITY ==========\n")


# ============================================================
# 5. PERSONALIZED RECOMMENDATION EVALUATION
# ============================================================

for index, property_row in sample_properties.iterrows():

    property_id = property_row["property_id"]

    neighbourhood = property_row["neighbourhood"]
    property_type = property_row["property_type"]
    room_type = property_row["room_type"]

    capacity = max(1, int(safe_number(property_row["capacity"], 1)))

    base_price = safe_number(
        property_row["nightly_price_npr"],
        1000
    )

    # Budget is intentionally higher than the reference
    # property's price to allow multiple possible matches.
    max_price = max(500, int(base_price * 1.5))

    preferences = {
        "neighbourhood": neighbourhood,
        "property_type": property_type,
        "room_type": room_type,
        "max_price": max_price,
        "guests": capacity,
    }

    execution = {"evaluation_type": "personalized", "scenario_id": index + 1, "reference_property_id": property_id}

    try:

        results = engine.recommend_personalized(
            preferences,
            number=10
        )

        if results.empty:
            execution.update(status="empty_results", result_count=0, error="No eligible recommendations")
            execution_results.append(execution)
            continue

        prices = pd.to_numeric(
            results["nightly_price_npr"],
            errors="coerce"
        )

        capacities = pd.to_numeric(
            results["capacity"],
            errors="coerce"
        )

        row = {
            "scenario_id": index + 1,
            "reference_property_id": property_id,
            "preferred_neighbourhood": neighbourhood,
            "preferred_property_type": property_type,
            "preferred_room_type": room_type,
            "guest_requirement": capacity,
            "maximum_budget_npr": max_price,
            "recommendation_count": len(results),
            "average_recommendation_score": get_score(results),
            "average_recommended_price_npr": safe_mean(prices),
            "budget_compliance_pct": round(
                (prices.dropna() <= max_price).mean() * 100, 2
            ) if prices.notna().any() else np.nan,
            "neighbourhood_match_pct": round(
                (results["neighbourhood"] == neighbourhood).mean() * 100, 2
            ),
            "property_type_match_pct": round(
                (results["property_type"] == property_type).mean() * 100, 2
            ),
            "room_type_match_pct": round(
                (results["room_type"] == room_type).mean() * 100, 2
            ),
            "guest_capacity_sufficiency_pct": round(
                (capacities >= capacity).mean() * 100, 2
            ) if capacities.notna().any() else np.nan,
        }

        add_result(scenario_results, row)
        execution.update(status="success", result_count=len(results), error="")
        execution_results.append(execution)

        # Collect recommendation component scores if available.
        score_columns = [
            "text_similarity",
            "neighbourhood_match",
            "budget_match",
            "property_type_match",
            "quality_score",
            "capacity_match",
        ]

        for column in score_columns:
            if column in results.columns:
                component_results.append({
                    "scenario_id": index + 1,
                    "component": column,
                    "average_component_score": safe_mean(results[column]),
                })

    except Exception as error:
        execution.update(status="failed", result_count=0, error=str(error))
        execution_results.append(execution)
        print(f"[WARNING] Scenario {index + 1} failed: {error}")


# ============================================================
# 6. SIMILAR PROPERTY EVALUATION
# ============================================================

print("\n========== SIMILAR PROPERTY QUALITY ==========\n")

for index, property_row in sample_properties.iterrows():

    property_id = property_row["property_id"]

    execution = {"evaluation_type": "similarity", "scenario_id": index + 1, "reference_property_id": property_id}

    try:

        results = engine.recommend_similar(
            property_id=property_id,
            number=10
        )

        if results.empty:
            execution.update(status="empty_results", result_count=0, error="No eligible similar properties")
            execution_results.append(execution)
            continue

        reference_price = safe_number(
            property_row["nightly_price_npr"],
            0
        )

        prices = pd.to_numeric(
            results["nightly_price_npr"],
            errors="coerce"
        )

        price_gaps = (
            (prices - reference_price).abs() / reference_price * 100
            if reference_price > 0
            else pd.Series(dtype=float)
        )

        row = {
            "reference_property_id": property_id,
            "reference_neighbourhood": property_row["neighbourhood"],
            "reference_property_type": property_row["property_type"],
            "reference_room_type": property_row["room_type"],
            "reference_price_npr": reference_price,
            "similar_property_count": len(results),
            "average_similarity_score": get_score(results),
            "same_neighbourhood_pct": round(
                (results["neighbourhood"] == property_row["neighbourhood"]).mean() * 100,
                2
            ),
            "same_property_type_pct": round(
                (results["property_type"] == property_row["property_type"]).mean() * 100,
                2
            ),
            "same_room_type_pct": round(
                (results["room_type"] == property_row["room_type"]).mean() * 100,
                2
            ),
            "average_relative_price_difference_pct": safe_mean(price_gaps),
        }

        similarity_results.append(row)
        execution.update(status="success", result_count=len(results), error="")
        execution_results.append(execution)

    except Exception as error:
        execution.update(status="failed", result_count=0, error=str(error))
        execution_results.append(execution)
        print(f"[WARNING] Similarity scenario {index + 1} failed: {error}")


# ============================================================
# 7. CREATE EVALUATION DATAFRAMES
# ============================================================

personalized_df = pd.DataFrame(scenario_results)
similarity_df = pd.DataFrame(similarity_results)
components_df = pd.DataFrame(component_results)
execution_df = pd.DataFrame(execution_results)


# ============================================================
# 8. PERSONALIZED QUALITY SUMMARY
# ============================================================

print("\n========== PERSONALIZED QUALITY SUMMARY ==========\n")

if not personalized_df.empty:

    personalized_summary = pd.DataFrame([
        {
            "metric": "Evaluation scenarios",
            "value": len(personalized_df),
        },
        {
            "metric": "Average recommendation score",
            "value": safe_mean(
                personalized_df["average_recommendation_score"]
            ),
        },
        {
            "metric": "Average budget compliance (%)",
            "value": safe_mean(
                personalized_df["budget_compliance_pct"]
            ),
        },
        {
            "metric": "Average neighbourhood match (%)",
            "value": safe_mean(
                personalized_df["neighbourhood_match_pct"]
            ),
        },
        {
            "metric": "Average property type match (%)",
            "value": safe_mean(
                personalized_df["property_type_match_pct"]
            ),
        },
        {
            "metric": "Average room type match (%)",
            "value": safe_mean(
                personalized_df["room_type_match_pct"]
            ),
        },
        {
            "metric": "Average guest capacity sufficiency (%)",
            "value": safe_mean(
                personalized_df["guest_capacity_sufficiency_pct"]
            ),
        },
        {
            "metric": "Average recommended price (NPR)",
            "value": safe_mean(
                personalized_df["average_recommended_price_npr"]
            ),
        },
    ])

    print(personalized_summary.to_string(index=False))

else:
    personalized_summary = pd.DataFrame()
    print("No personalized evaluation results generated.")


# ============================================================
# 9. SIMILAR PROPERTY QUALITY SUMMARY
# ============================================================

print("\n========== SIMILAR PROPERTY QUALITY SUMMARY ==========\n")

if not similarity_df.empty:

    similarity_summary = pd.DataFrame([
        {
            "metric": "Evaluation scenarios",
            "value": len(similarity_df),
        },
        {
            "metric": "Average similarity score",
            "value": safe_mean(
                similarity_df["average_similarity_score"]
            ),
        },
        {
            "metric": "Same neighbourhood (%)",
            "value": safe_mean(
                similarity_df["same_neighbourhood_pct"]
            ),
        },
        {
            "metric": "Same property type (%)",
            "value": safe_mean(
                similarity_df["same_property_type_pct"]
            ),
        },
        {
            "metric": "Same room type (%)",
            "value": safe_mean(
                similarity_df["same_room_type_pct"]
            ),
        },
        {
            "metric": "Average relative price difference (%)",
            "value": safe_mean(
                similarity_df["average_relative_price_difference_pct"]
            ),
        },
    ])

    print(similarity_summary.to_string(index=False))

else:
    similarity_summary = pd.DataFrame()
    print("No similar-property evaluation results generated.")


# ============================================================
# 10. RECOMMENDATION SCORE DISTRIBUTION
# ============================================================

print("\n========== RECOMMENDATION SCORE DISTRIBUTION ==========\n")

score_distribution = pd.DataFrame()

if not personalized_df.empty:

    scores = pd.to_numeric(
        personalized_df["average_recommendation_score"],
        errors="coerce"
    ).dropna()

    if not scores.empty:

        score_distribution = pd.DataFrame([
            {"statistic": "Minimum", "score": scores.min()},
            {"statistic": "25th percentile", "score": scores.quantile(0.25)},
            {"statistic": "Median", "score": scores.median()},
            {"statistic": "75th percentile", "score": scores.quantile(0.75)},
            {"statistic": "Maximum", "score": scores.max()},
            {"statistic": "Mean", "score": scores.mean()},
        ])

        print(score_distribution.to_string(index=False))


# ============================================================
# 11. SAVE EVALUATION REPORTS
# ============================================================

reports = {
    "recommendation_personalized_quality.csv": personalized_df,
    "recommendation_similarity_quality.csv": similarity_df,
    "recommendation_component_analysis.csv": components_df,
    "recommendation_personalized_summary.csv": personalized_summary,
    "recommendation_similarity_summary.csv": similarity_summary,
    "recommendation_score_distribution.csv": score_distribution,
    "recommendation_evaluation_execution.csv": execution_df,
}

print("\n========== SAVING EVALUATION REPORTS ==========\n")

for filename, dataframe in reports.items():

    output_path = REPORT_DIR / filename

    dataframe.to_csv(output_path, index=False)

    print(f"[SAVED] {output_path}")


# ============================================================
# 12. FINAL EVALUATION SUMMARY
# ============================================================

print("\n========== FINAL QUALITY EVALUATION ==========\n")

print(f"Personalized scenarios evaluated: {len(personalized_df)}")
print(f"Similarity scenarios evaluated: {len(similarity_df)}")
if not execution_df.empty:
    print("Scenario execution status:")
    print(execution_df.groupby(["evaluation_type", "status"]).size().to_string())
    failed_count = int(execution_df["status"].eq("failed").sum())
    if failed_count:
        print(f"\n[ERROR] {failed_count} evaluation scenario(s) failed. See recommendation_evaluation_execution.csv.")

if not personalized_df.empty:

    print(
        "Average personalized recommendation score:",
        f"{safe_mean(personalized_df['average_recommendation_score'])}%"
    )

    print(
        "Average budget compliance:",
        f"{safe_mean(personalized_df['budget_compliance_pct'])}%"
    )

if not similarity_df.empty:

    print(
        "Average similar-property score:",
        f"{safe_mean(similarity_df['average_similarity_score'])}%"
    )

print("\nIMPORTANT:")
print(
    "These are heuristic relevance and preference-match diagnostics. "
    "They are not ground-truth accuracy, precision, recall, or user satisfaction."
)

if execution_df.empty or not execution_df["status"].eq("success").any():
    raise RuntimeError("No recommendation evaluation scenarios completed successfully.")

if execution_df["status"].eq("failed").any():
    raise SystemExit(1)

print("\nSUCCESS: Recommendation quality evaluation completed.")
