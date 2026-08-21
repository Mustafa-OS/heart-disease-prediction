"""Data cleaning for the Cleveland heart-disease dataset.

Reproduces the cleaning described in the group methodology (report section
3.2.1). The raw Kaggle file (data/heart.csv, 1025 rows) is the UCI Cleveland
dataset with heavy duplication, a handful of invalid category codes, and a
target column whose labels are inverted relative to the Kaggle documentation.

Steps:
  1. Remove exact duplicate records (duplicates would inflate performance by
     leaking identical patients across the train/validation/test splits).
  2. Remove rows with out-of-range category codes:
       - thal: valid codes are 1-3; thal = 0 marks a missing value.
       - ca:   valid counts are 0-3; ca = 4 marks a missing value.
  3. Fix the inverted target column: in the raw file the 0/1 labels are
     swapped relative to the documentation on Kaggle, so they are corrected
     here (after the fix, target = 1 means heart disease is present).

Writes the cleaned dataset to data/heart_cleaned.csv.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "heart.csv"
CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_cleaned.csv"


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"Raw dataset: {len(df)} rows")

    # 1. Remove exact duplicate records
    df = df.drop_duplicates()
    print(f"After removing exact duplicates: {len(df)} rows")

    # 2. Remove rows with out-of-range codes for thal and ca
    df = df[df["thal"].between(1, 3) & df["ca"].between(0, 3)]
    print(f"After removing out-of-range thal/ca codes: {len(df)} rows")

    # 3. Fix the inverted target column (swap the 0s and 1s)
    df = df.copy()
    df["target"] = 1 - df["target"]
    print(f"Disease rate after target fix: {df['target'].mean():.1%}")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset written to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
