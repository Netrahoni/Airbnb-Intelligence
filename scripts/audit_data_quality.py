"""Create a reproducible quality profile for processed CSV datasets."""

from pathlib import Path
import argparse

import pandas as pd

from src.data_quality import build_data_quality_report


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a failing exit code when a processed dataset is empty.",
    )
    args = parser.parse_args()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {DATA_DIR}")

    datasets = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        options = {}
        if path.name == "properties_clean.csv":
            options["key_columns"] = ("property_id",)
            options["numeric_bounds"] = {
                "nightly_price_npr": (0, None),
                "rating": (0, 5),
                "review_count": (0, None),
                "capacity": (0, None),
            }
        elif path.name in {"recommendation_features.csv"}:
            options["key_columns"] = ("property_id",)
            options["numeric_bounds"] = {
                "nightly_price_npr": (0, None),
                "rating": (0, 5),
                "review_count": (0, None),
                "capacity": (0, None),
            }
        elif path.name in {"price_prediction_features.csv"}:
            options["numeric_bounds"] = {
                "base_price_npr": (0, None),
                "lead_time_days": (0, None),
                "capacity": (0, None),
            }
        datasets.append((path.name, frame, options))

    if not datasets:
        raise FileNotFoundError(f"No processed CSV datasets found in {DATA_DIR}")

    report = build_data_quality_report(datasets)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "data_quality_report.csv"
    report.to_csv(output, index=False)

    print(f"Profiled {len(datasets)} datasets and {len(report)} columns.")
    print(f"Quality report: {output}")
    print("\nHighest missing-value columns:")
    print(
        report.sort_values(
            ["missing_percent", "missing_count"],
            ascending=False,
        ).head(15).to_string(index=False)
    )

    empty_datasets = sorted(
        name for name, frame, _ in datasets if frame.empty
    )
    duplicate_properties = report.loc[
        report["dataset"].isin(["properties_clean.csv", "recommendation_features.csv"]),
        ["dataset", "duplicate_key_rows"],
    ].drop_duplicates()
    duplicate_properties = duplicate_properties[
        duplicate_properties["duplicate_key_rows"] > 0
    ]

    if args.check and (empty_datasets or not duplicate_properties.empty):
        if empty_datasets:
            print("\nERROR: Empty processed datasets:", ", ".join(empty_datasets))
        if not duplicate_properties.empty:
            print("\nERROR: Duplicate property IDs detected:")
            print(duplicate_properties.to_string(index=False))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
