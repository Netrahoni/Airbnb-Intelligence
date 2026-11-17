
import pandas as pd
from pathlib import Path

print("========== PRICE PREDICTION ERROR ANALYSIS ==========")

# Load predictions
file_path = Path("reports/price_predictions.csv")
df = pd.read_csv(file_path)

# Calculate additional error measurements
df["signed_error_npr"] = (
    df["predicted_price_npr"] - df["actual_price_npr"]
)

df["percentage_error"] = (
    df["absolute_error_npr"] /
    df["actual_price_npr"].replace(0, pd.NA)
) * 100

# Categorize actual prices
bins = [0, 2000, 4000, 6000, float("inf")]

labels = [
    "Budget (up to 2000)",
    "Mid-range (2001-4000)",
    "Premium (4001-6000)",
    "Luxury (above 6000)"
]

df["price_category"] = pd.cut(
    df["actual_price_npr"],
    bins=bins,
    labels=labels
)

# Analyze each price category
analysis = df.groupby(
    "price_category",
    observed=False
).agg(
    sample_count=("actual_price_npr", "count"),
    average_actual_price=("actual_price_npr", "mean"),
    average_predicted_price=("predicted_price_npr", "mean"),
    mean_absolute_error=("absolute_error_npr", "mean"),
    mean_percentage_error=("percentage_error", "mean"),
    average_signed_error=("signed_error_npr", "mean")
).round(2)

print("\n========== ERROR BY PRICE CATEGORY ==========")
print(analysis.to_string())

# Find the 10 largest prediction errors
worst_predictions = df.nlargest(
    10, "absolute_error_npr"
)

print("\n========== 10 LARGEST PREDICTION ERRORS ==========")
print(
    worst_predictions[
        [
            "actual_price_npr",
            "predicted_price_npr",
            "absolute_error_npr",
            "signed_error_npr"
        ]
    ].round(2).to_string(index=False)
)

# Save reports
analysis.to_csv(
    "reports/price_error_by_category.csv"
)

df.to_csv(
    "reports/price_error_analysis.csv",
    index=False
)

print("\nReports saved successfully:")
print("reports/price_error_by_category.csv")
print("reports/price_error_analysis.csv")

print("\nError analysis completed!")
