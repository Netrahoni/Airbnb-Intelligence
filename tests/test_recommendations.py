
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advanced_recommender import AdvancedPropertyRecommender

DATA_PATH = ROOT / "data" / "processed" / "properties_clean.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA AND INITIALIZE ENGINE
# ============================================================

print("\n========== RECOMMENDATION MODEL EVALUATION ==========\n")

properties = pd.read_csv(DATA_PATH)

print("Dataset shape:", properties.shape)

engine = AdvancedPropertyRecommender(properties)

print("Recommendation engine initialized.")
print("Published properties:", len(engine.properties))


# ============================================================
# 3. TEST RESULT MANAGEMENT
# ============================================================

test_results = []


def run_test(test_name, test_function):

    try:
        details = test_function()

        status = "PASS"

    except Exception as error:

        status = "FAIL"
        details = str(error)

    test_results.append({
        "test_name": test_name,
        "status": status,
        "details": str(details),
    })

    symbol = "PASS" if status == "PASS" else "FAIL"

    print(f"[{symbol}] {test_name}")
    print("       ", details)


# ============================================================
# 4. PERSONALIZED RECOMMENDATION TESTS
# ============================================================

personalized_results = None


def test_personalized_recommendations():

    global personalized_results

    preferences = {
        "max_price": 5000,
        "guests": 2,
    }

    personalized_results = engine.recommend_personalized(
        preferences,
        number=10,
    )

    assert not personalized_results.empty, (
        "No personalized recommendations were returned."
    )

    return f"{len(personalized_results)} recommendations generated."


def test_budget_constraint():

    assert personalized_results is not None

    assert (
        personalized_results["nightly_price_npr"] <= 5000
    ).all(), "A recommended property exceeded the budget."

    return "All recommendations are within NPR 5,000."


def test_guest_capacity_constraint():

    assert personalized_results is not None

    assert (
        personalized_results["capacity"] >= 2
    ).all(), "A recommended property has insufficient capacity for 2 guests."

    return "All recommendations satisfy the minimum guest capacity of 2."


def test_recommendation_limit():

    assert len(personalized_results) <= 10

    return f"Returned {len(personalized_results)} of maximum 10."


def test_duplicate_recommendations():

    duplicates = personalized_results["property_id"].duplicated().sum()

    assert duplicates == 0, (
        f"Found {duplicates} duplicate property IDs."
    )

    return "No duplicate property recommendations."


def test_score_order():

    scores = personalized_results["Recommendation Score (%)"]

    assert scores.is_monotonic_decreasing, (
        "Recommendations are not sorted by score."
    )

    return "Recommendation scores are sorted from highest to lowest."


def test_valid_prices():

    assert (
        personalized_results["nightly_price_npr"] > 0
    ).all(), "Invalid or zero prices found."

    return "All recommended properties have positive prices."


# ============================================================
# 5. SIMILAR PROPERTY TESTS
# ============================================================

selected_property_id = str(
    engine.properties.iloc[0]["property_id"]
)

similar_results = None


def test_similar_recommendations():

    global similar_results

    similar_results = engine.recommend_similar(
        property_id=selected_property_id,
        number=5,
    )

    assert not similar_results.empty, (
        "No similar properties were returned."
    )

    return (
        f"{len(similar_results)} similar properties generated "
        f"for {selected_property_id}."
    )


def test_reference_exclusion():

    assert similar_results is not None

    assert (
        similar_results["property_id"].astype(str)
        != selected_property_id
    ).all(), "The original property appeared in its own recommendations."

    return "Original property correctly excluded."


def test_similar_score_order():

    scores = similar_results["Recommendation Score (%)"]

    assert scores.is_monotonic_decreasing, (
        "Similar properties are not sorted by score."
    )

    return "Similar property scores are correctly ordered."


# ============================================================
# 6. FILTER VALIDATION TESTS
# ============================================================

def test_minimum_rating_filter():

    results = engine.recommend_personalized(
        {},
        number=10,
        min_rating=4.0,
    )

    assert (
        results["rating"] >= 4.0
    ).all(), "A property below the minimum rating was returned."

    return f"{len(results)} results satisfy the 4.0 minimum rating."


def test_similar_property_budget_filter():

    results = engine.recommend_similar(
        property_id=selected_property_id,
        number=5,
        max_budget=5000,
    )

    assert (
        results["nightly_price_npr"] <= 5000
    ).all(), "Similar-property results exceeded the budget."

    return f"{len(results)} results satisfy the NPR 5,000 budget."


# ============================================================
# 7. INVALID INPUT TESTS
# ============================================================

def test_invalid_budget():

    try:

        engine.recommend_personalized(
            {"max_price": 0},
            number=5,
        )

    except ValueError:

        return "Invalid zero budget correctly rejected."

    raise AssertionError("Zero budget was accepted.")


def test_invalid_guest_count():

    try:

        engine.recommend_personalized(
            {"guests": 0},
            number=5,
        )

    except ValueError:

        return "Invalid guest count correctly rejected."

    raise AssertionError("Zero guest count was accepted.")


def test_invalid_recommendation_count():

    try:

        engine.recommend_personalized(
            {},
            number=0,
        )

    except ValueError:

        return "Invalid recommendation count correctly rejected."

    raise AssertionError("Zero recommendation count was accepted.")


def test_unknown_property():

    try:

        engine.recommend_similar(
            property_id="UNKNOWN_PROPERTY_ID",
            number=5,
        )

    except ValueError:

        return "Unknown property ID correctly rejected."

    raise AssertionError("Unknown property ID was accepted.")


# ============================================================
# 8. EXECUTE ALL TESTS
# ============================================================

print("\n========== PERSONALIZED RECOMMENDATION TESTS ==========\n")

run_test(
    "Personalized recommendation generation",
    test_personalized_recommendations,
)

run_test(
    "Strict budget constraint",
    test_budget_constraint,
)
run_test(
    "Strict guest capacity constraint",
    test_guest_capacity_constraint,
)

run_test(
    "Recommendation count limit",
    test_recommendation_limit,
)

run_test(
    "Duplicate property detection",
    test_duplicate_recommendations,
)

run_test(
    "Recommendation score ordering",
    test_score_order,
)

run_test(
    "Positive property prices",
    test_valid_prices,
)


print("\n========== SIMILAR PROPERTY TESTS ==========\n")

run_test(
    "Similar property generation",
    test_similar_recommendations,
)

run_test(
    "Reference property exclusion",
    test_reference_exclusion,
)

run_test(
    "Similar property score ordering",
    test_similar_score_order,
)


print("\n========== FILTER VALIDATION TESTS ==========\n")

run_test(
    "Minimum rating constraint",
    test_minimum_rating_filter,
)

run_test(
    "Similar property budget constraint",
    test_similar_property_budget_filter,
)


print("\n========== INVALID INPUT TESTS ==========\n")

run_test(
    "Invalid budget handling",
    test_invalid_budget,
)

run_test(
    "Invalid guest count handling",
    test_invalid_guest_count,
)

run_test(
    "Invalid recommendation count handling",
    test_invalid_recommendation_count,
)

run_test(
    "Unknown property handling",
    test_unknown_property,
)


# ============================================================
# 9. RECOMMENDATION QUALITY DIAGNOSTICS
# ============================================================

print("\n========== RECOMMENDATION QUALITY DIAGNOSTICS ==========\n")

if personalized_results is not None and not personalized_results.empty:

    average_score = personalized_results[
        "Recommendation Score (%)"
    ].mean()

    average_price = personalized_results[
        "nightly_price_npr"
    ].mean()

    guest_capacity_fit = (
        personalized_results["capacity"] >= 2
    ).sum()

    print("Average recommendation score:", round(average_score, 2), "%")

    print("Average recommended price: NPR", round(average_price, 2))

    print(
        "Properties with capacity for 2 guests:",
        guest_capacity_fit,
        "out of",
        len(personalized_results),
    )

    print(
    "Capacity note: Guest capacity is now a strict filtering constraint."
)


# ============================================================
# 10. SAVE EVALUATION REPORT
# ============================================================

evaluation_df = pd.DataFrame(test_results)

report_path = REPORT_DIR / "recommendation_evaluation.csv"

evaluation_df.to_csv(report_path, index=False)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

passed = (evaluation_df["status"] == "PASS").sum()

failed = (evaluation_df["status"] == "FAIL").sum()

total = len(evaluation_df)

success_rate = (passed / total * 100) if total else 0

print("\n========== FINAL EVALUATION SUMMARY ==========\n")

print("Total tests:", total)
print("Passed:", passed)
print("Failed:", failed)
print("Success rate:", f"{success_rate:.2f}%")

print("\nEvaluation report saved:")
print(report_path)

if failed == 0:

    print("\nSUCCESS: All recommendation tests passed!")

else:

    print("\nWARNING: Some recommendation tests failed.")
    print("Review the failed tests before modifying the model.")
