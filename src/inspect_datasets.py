import pandas as pd
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

# Find project root automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = PROJECT_ROOT / "data" / "processed"
REPORT_FOLDER = PROJECT_ROOT / "reports"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

report = StringIO()
dataset_summary = []

with redirect_stdout(report):

    print("=" * 75)
    print("KATHMANDU AIRBNB ML ANALYSIS")
    print("DATASET INSPECTION REPORT")
    print("=" * 75)

    for filepath in sorted(DATA_FOLDER.glob("*.csv")):

        print("\n\n" + "=" * 75)
        print("DATASET:", filepath.name)
        print("=" * 75)

        try:
            df = pd.read_csv(filepath)

            print("\nDataset Shape:", df.shape)

            print("\nColumn Names:")
            print(df.columns.tolist())

            print("\nData Types:")
            print(df.dtypes.to_string())

            print("\nMissing Values:")
            print(df.isnull().sum().to_string())

            print("\nMissing Percentage:")
            print((df.isnull().mean() * 100).round(2).to_string())

            print("\nDuplicate Records:")
            print(df.duplicated().sum())

            print("\nFirst 3 Records:")
            print(df.head(3).to_string())

            print("\nNumerical Statistics:")
            print(df.describe().to_string())

            dataset_summary.append(
                (filepath.name, df.shape[0], df.shape[1], "SUCCESS")
            )

        except Exception as error:
            print("ERROR:", str(error))
            dataset_summary.append(
                (filepath.name, 0, 0, "FAILED")
            )

    print("\n\n" + "=" * 75)
    print("INSPECTION COMPLETED")
    print("=" * 75)

# Save complete report
report_path = REPORT_FOLDER / "dataset_inspection.txt"
report_path.write_text(report.getvalue(), encoding="utf-8")

print("\nDATASET INSPECTION COMPLETED SUCCESSFULLY")

for name, rows, columns, status in dataset_summary:
    print(f"{name}: {rows} rows, {columns} columns [{status}]")

print("\nReport saved at:", report_path)
print("Report size:", report_path.stat().st_size, "bytes")
