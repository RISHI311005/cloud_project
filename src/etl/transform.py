"""ETL transform script for food waste dataset."""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CATEGORICAL_COLUMNS: List[str] = [
    "meal_type",
    "weather",
    "menu_type",
    "food_category",
]

NUMERIC_COLUMNS: List[str] = [
    "expected_people",
    "actual_people",
    "quantity_prepared_kg",
    "quantity_consumed_kg",
    "leftover_kg",
]


def load_dataset(input_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(input_path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode/Unknown for categorical."""
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            if df[column].dropna().empty:
                df[column] = df[column].fillna("Unknown")
            else:
                df[column] = df[column].fillna(df[column].mode()[0])

    # Basic binary columns, default to 0 if missing
    for column in ["is_holiday", "special_event"]:
        if column in df.columns:
            df[column] = df[column].fillna(0).astype(int)

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date and add month, day, and weekday index."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday_index"] = df["date"].dt.weekday
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create attendance_ratio and waste_percentage features."""
    df = df.copy()
    df["attendance_ratio"] = df["actual_people"] / df["expected_people"].replace(0, np.nan)
    df["waste_percentage"] = df["leftover_kg"] / df["quantity_prepared_kg"].replace(0, np.nan)

    df["attendance_ratio"] = df["attendance_ratio"].fillna(0)
    df["waste_percentage"] = df["waste_percentage"].fillna(0)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical variables."""
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)


def scale_numeric_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Scale numeric features using StandardScaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def transform_data(input_path: str, output_path: str, scale_features: bool) -> None:
    """Run full ETL transform and save output to CSV."""
    df = load_dataset(input_path)
    df = handle_missing_values(df)
    df = add_date_features(df)
    df = add_engineered_features(df)
    df = encode_categoricals(df)

    if scale_features:
        scale_columns = [
            "expected_people",
            "actual_people",
            "quantity_prepared_kg",
            "quantity_consumed_kg",
            "leftover_kg",
            "attendance_ratio",
            "waste_percentage",
            "month",
            "day",
            "weekday_index",
        ]
        scale_columns = [col for col in scale_columns if col in df.columns]
        df = scale_numeric_features(df, scale_columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transform food waste dataset.")
    parser.add_argument(
        "--input",
        default="data/raw/food_waste_dataset.csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/food_waste_dataset_processed.csv",
        help="Path to save the processed CSV file.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale numeric features using StandardScaler.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transform_data(args.input, args.output, args.scale)
