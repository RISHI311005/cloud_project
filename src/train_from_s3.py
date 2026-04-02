"""Train food waste models from S3 and upload the best model back to S3."""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from typing import Dict, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def read_csv_from_s3(bucket: str, key: str, region: str) -> pd.DataFrame:
    """Read a CSV file from S3 into a DataFrame."""
    s3 = boto3.client("s3", region_name=region)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise RuntimeError(f"Failed to read s3://{bucket}/{key}: {exc}") from exc

    return pd.read_csv(obj["Body"])


def upload_file_to_s3(bucket: str, key: str, local_path: str, region: str) -> None:
    """Upload a local file to S3."""
    s3 = boto3.client("s3", region_name=region)
    try:
        with open(local_path, "rb") as file_handle:
            s3.put_object(Bucket=bucket, Key=key, Body=file_handle)
    except ClientError as exc:
        raise RuntimeError(f"Failed to upload to s3://{bucket}/{key}: {exc}") from exc


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features (X) and target (y)."""
    df = df.copy()
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if "day_of_week" in df.columns:
        df = df.drop(columns=["day_of_week"])

    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Return MAE, RMSE, and R2 metrics for a model."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_models(X_train, y_train) -> Dict[str, object]:
    """Train three regression models."""
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """Select best model by lowest RMSE."""
    return min(results, key=lambda name: results[name]["rmse"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train models from S3 data and upload best model.")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET"), help="S3 bucket name")
    parser.add_argument(
        "--input-key",
        default=os.getenv("S3_PROCESSED_KEY", "processed/cleaned_data.csv"),
        help="S3 key for processed CSV",
    )
    parser.add_argument(
        "--output-key",
        default=os.getenv("S3_MODEL_KEY", "models/model.pkl"),
        help="S3 key to upload model",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        help="AWS region",
    )
    parser.add_argument(
        "--local-model-path",
        default="model/model.pkl",
        help="Local path to save the model",
    )
    return parser.parse_args()


def main() -> None:
    """Train models from S3 data and upload the best model."""
    args = parse_args()

    if not args.bucket:
        raise ValueError("S3 bucket is required. Use --bucket or set S3_BUCKET.")

    print(f"Loading dataset from s3://{args.bucket}/{args.input_key}")
    df = read_csv_from_s3(args.bucket, args.input_key, args.region)

    X, y = split_features_target(df, target="leftover_kg")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}

    print("Model Performance (lower MAE/RMSE is better, higher R2 is better)")
    for name, metrics in results.items():
        print(
            f"- {name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}"
        )

    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name}")

    os.makedirs(os.path.dirname(args.local_model_path), exist_ok=True)
    joblib.dump(best_model, args.local_model_path)
    print(f"Saved model locally to {args.local_model_path}")

    print(f"Uploading model to s3://{args.bucket}/{args.output_key}")
    upload_file_to_s3(args.bucket, args.output_key, args.local_model_path, args.region)
    print("Upload complete.")


if __name__ == "__main__":
    main()
