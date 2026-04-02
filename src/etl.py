"""ETL pipeline for the food waste dataset (local or S3)."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _read_from_s3(bucket: str, key: str, region: str) -> pd.DataFrame:
    """Read a CSV file from S3 into a DataFrame."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def _write_to_s3(df: pd.DataFrame, bucket: str, key: str, region: str) -> None:
    """Write a DataFrame as CSV to S3."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def run_etl(
    input_path: str = "data/raw/food_waste_dataset.csv",
    output_path: str = "data/processed/cleaned_data.csv",
    scale_features: bool = True,
) -> None:
    """Load, clean, engineer features, encode, scale, and save dataset."""
    use_s3 = os.getenv("USE_S3", "false").lower() == "true"
    if use_s3:
        bucket = os.environ["S3_BUCKET"]
        raw_key = os.getenv("S3_RAW_KEY", "raw/food_waste_dataset.csv")
        region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
        df = _read_from_s3(bucket, raw_key, region)
    else:
        df = pd.read_csv(input_path)

    # Handle missing values
    numeric_cols = [
        "expected_people",
        "actual_people",
        "quantity_prepared_kg",
        "quantity_consumed_kg",
        "leftover_kg",
    ]
    categorical_cols = ["meal_type", "weather", "menu_type", "food_category"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    for col in ["is_holiday", "special_event"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Date features
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday_index"] = df["date"].dt.weekday

    # Engineered features
    df["attendance_ratio"] = df["actual_people"] / df["expected_people"].replace(0, np.nan)
    df["waste_percentage"] = df["leftover_kg"] / df["quantity_prepared_kg"].replace(0, np.nan)
    df["attendance_ratio"] = df["attendance_ratio"].fillna(0)
    df["waste_percentage"] = df["waste_percentage"].fillna(0)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Optional scaling
    if scale_features:
        scale_cols = [
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
        scale_cols = [c for c in scale_cols if c in df.columns]
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    if use_s3:
        bucket = os.environ["S3_BUCKET"]
        processed_key = os.getenv("S3_PROCESSED_KEY", "processed/cleaned_data.csv")
        region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
        _write_to_s3(df, bucket, processed_key, region)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    run_etl()
