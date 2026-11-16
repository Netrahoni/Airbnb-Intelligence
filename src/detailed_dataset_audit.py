from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        print(f"\nFILE NOT FOUND: {filename}")
        return None
    return pd.read_csv(path)

properties = load_csv("properties_clean.csv")
availability = load_csv("availability_clean.csv")
prices = load_csv("price_observations_clean.csv")
bookings = load_csv("booking_cancellations_clean.csv")
tourism = load_csv("nepal_monthly_tourist_arrivals_clean.csv")

results = []

def add_result(section, metric, value):
    results.append({
        "section": section,
        "metric": metric,
        "value": str(value)
    })

print("\n========== DETAILED DATASET AUDIT ==========")

# 1. Properties
if properties is not None:
    add_result("Properties", "Total rows", len(properties))
    add_result("Properties", "Unique property IDs",
               properties["property_id"].nunique())

    if "source_name" in properties.columns:
        print("\nProperty source names:")
        print(properties["source_name"].value_counts(dropna=False))
        add_result("Properties", "Source counts",
                   properties["source_name"].value_counts(dropna=False).to_dict())

    if "listing_url" in properties.columns:
        add_result("Properties", "Missing listing URLs",
                   properties["listing_url"].isna().sum())

# 2. Availability
if availability is not None:
    availability["date"] = pd.to_datetime(
        availability["date"], errors="coerce"
    )

    counts = availability.groupby("property_id").size()

    add_result("Availability", "Total rows", len(availability))
    add_result("Availability", "Unique properties",
               availability["property_id"].nunique())
    add_result("Availability", "Minimum records per property",
               counts.min())
    add_result("Availability", "Maximum records per property",
               counts.max())
    add_result("Availability", "Properties with exactly 90 records",
               (counts == 90).sum())
    add_result("Availability", "Date range",
               f"{availability['date'].min()} to {availability['date'].max()}")

    date_counts = availability.groupby("property_id")["date"].agg(
        ["min", "max", "nunique"]
    )

    add_result("Availability", "Properties with same date range",
               ((date_counts["min"] == date_counts["min"].iloc[0]) &
                (date_counts["max"] == date_counts["max"].iloc[0])).sum())

    add_result("Availability", "Invalid dates",
               availability["date"].isna().sum())

# 3. Price observations
if prices is not None:
    prices["observation_date"] = pd.to_datetime(
        prices["observation_date"], errors="coerce"
    )
    prices["stay_date"] = pd.to_datetime(
        prices["stay_date"], errors="coerce"
    )

    price_counts = prices.groupby("property_id").size()

    add_result("Prices", "Total rows", len(prices))
    add_result("Prices", "Unique properties",
               prices["property_id"].nunique())
    add_result("Prices", "Minimum observations per property",
               price_counts.min())
    add_result("Prices", "Maximum observations per property",
               price_counts.max())
    add_result("Prices", "Observation date range",
               f"{prices['observation_date'].min()} to {prices['observation_date'].max()}")
    add_result("Prices", "Stay date range",
               f"{prices['stay_date'].min()} to {prices['stay_date'].max()}")

    if "source_name" in prices.columns:
        print("\nPrice source names:")
        print(prices["source_name"].value_counts(dropna=False))
        add_result("Prices", "Source counts",
                   prices["source_name"].value_counts(dropna=False).to_dict())

    if "source_url" in prices.columns:
        add_result("Prices", "Missing source URLs",
                   prices["source_url"].isna().sum())

# 4. Booking cancellations — separate benchmark dataset
if bookings is not None:
    add_result("Bookings", "Total rows", len(bookings))

    if "adr" in bookings.columns:
        add_result("Bookings", "Negative ADR rows",
                   (bookings["adr"] < 0).sum())
        add_result("Bookings", "Zero ADR rows",
                   (bookings["adr"] == 0).sum())

    if "guest_count_is_zero" in bookings.columns:
        add_result("Bookings", "Zero guest rows",
                   bookings["guest_count_is_zero"].sum())

    if "hotel" in bookings.columns:
        add_result("Bookings", "Hotel categories",
                   bookings["hotel"].value_counts().to_dict())

    if "arrival_date" in bookings.columns:
        add_result("Bookings", "Arrival date examples",
                   bookings["arrival_date"].head(5).tolist())

# 5. Tourist arrivals
if tourism is not None:
    add_result("Tourism", "Total rows", len(tourism))

    if "tourist_arrivals" in tourism.columns:
        lowest = tourism.loc[tourism["tourist_arrivals"].idxmin()]

        add_result("Tourism", "Minimum arrivals",
                   lowest["tourist_arrivals"])
        add_result("Tourism", "Month with minimum arrivals",
                   lowest["month_date"])

    if "source_name" in tourism.columns:
        add_result("Tourism", "Source names",
                   tourism["source_name"].value_counts().to_dict())

# Display and save
report = pd.DataFrame(results)

print("\n========== AUDIT RESULTS ==========")
print(report.to_string(index=False))

output = REPORTS / "detailed_dataset_audit.csv"
report.to_csv(output, index=False)

print(f"\nAudit report saved to: {output}")
