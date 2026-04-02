"""Flask app for food waste prediction."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List

import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

import boto3
from botocore.exceptions import ClientError

MODEL_PATH = os.path.join("model", "model.pkl")
RAW_DATA_PATH = os.path.join("data", "raw", "food_waste_dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "models/model.pkl")
S3_RAW_KEY = os.getenv("S3_RAW_KEY", "raw/food_waste_dataset.csv")
S3_PROCESSED_KEY = os.getenv("S3_PROCESSED_KEY", "processed/cleaned_data.csv")
S3_PREDICTIONS_PREFIX = os.getenv("S3_PREDICTIONS_PREFIX", "predictions/")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")

CATEGORICAL_COLUMNS: List[str] = [
    "meal_type",
    "weather",
    "menu_type",
    "food_category",
]

SCALE_COLUMNS: List[str] = [
    "expected_people",
    "actual_people",
    "quantity_prepared_kg",
    "quantity_consumed_kg",
    "attendance_ratio",
    "waste_percentage",
    "month",
    "day",
    "weekday_index",
]

app = Flask(__name__, template_folder="../templates")


def _read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read a CSV file from S3 into a DataFrame."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise RuntimeError(f"Failed to read s3://{bucket}/{key}: {exc}") from exc

    return pd.read_csv(obj["Body"])


def _load_model_from_s3(bucket: str, key: str):
    """Load a joblib model stored in S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise RuntimeError(f"Failed to read s3://{bucket}/{key}: {exc}") from exc

    buffer = io.BytesIO(obj["Body"].
                        read())
    return joblib.load(buffer)


def _upload_prediction_log(bucket: str, payload: dict) -> None:
    """Upload a prediction log as JSON into the predictions/ folder in S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    key = f"{S3_PREDICTIONS_PREFIX.rstrip('/')}/prediction_{timestamp}.json"
    body = json.dumps(payload, indent=2)

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as exc:
        print(f"Warning: failed to upload prediction log to s3://{bucket}/{key}: {exc}")


def estimate_per_person_kg(meal_type: str, menu_type: str) -> float:
    """Estimate per-person quantity based on meal and menu."""
    base = 0.35 if meal_type == "breakfast" else 0.5 if meal_type == "lunch" else 0.45
    if menu_type == "premium":
        base += 0.05
    if menu_type == "vegetarian":
        base -= 0.03
    return max(base, 0.2)


def build_feature_frame(form_data: dict) -> pd.DataFrame:
    """Create a single-row dataframe matching training features."""
    today = datetime.utcnow()

    meal_type = form_data["meal_type"]
    menu_type = form_data["menu_type"]

    expected_people = float(form_data["expected_people"])
    actual_people = float(form_data["actual_people"])
    quantity_prepared = float(form_data["quantity_prepared_kg"])

    per_person_kg = estimate_per_person_kg(meal_type, menu_type)
    estimated_consumed = min(quantity_prepared, actual_people * per_person_kg)

    attendance_ratio = actual_people / expected_people if expected_people > 0 else 0
    waste_percentage = (
        (quantity_prepared - estimated_consumed) / quantity_prepared if quantity_prepared > 0 else 0
    )

    row = {
        "date": today.strftime("%Y-%m-%d"),
        "day_of_week": today.strftime("%A"),
        "meal_type": meal_type,
        "is_holiday": int(form_data["is_holiday"]),
        "special_event": int(form_data["special_event"]),
        "weather": form_data["weather"],
        "expected_people": expected_people,
        "actual_people": actual_people,
        "quantity_prepared_kg": quantity_prepared,
        "quantity_consumed_kg": estimated_consumed,
        "leftover_kg": 0.0,
        "menu_type": menu_type,
        "food_category": form_data["food_category"],
        "month": today.month,
        "day": today.day,
        "weekday_index": today.weekday(),
        "attendance_ratio": attendance_ratio,
        "waste_percentage": waste_percentage,
    }

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)

    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if "day_of_week" in df.columns:
        df = df.drop(columns=["day_of_week"])

    return df


def load_training_columns() -> List[str]:
    """Load training feature columns from processed dataset."""
    if USE_S3:
        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET is required when USE_S3=true.")
        df = _read_csv_from_s3(S3_BUCKET, S3_PROCESSED_KEY)
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.drop(columns=["leftover_kg"], errors="ignore")
    df = df.drop(columns=["date", "day_of_week"], errors="ignore")
    return df.columns.tolist()


def build_scaler() -> StandardScaler:
    """Fit a scaler based on raw data preprocessing."""
    if USE_S3:
        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET is required when USE_S3=true.")
        df = _read_csv_from_s3(S3_BUCKET, S3_RAW_KEY)
    else:
        df = pd.read_csv(RAW_DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday_index"] = df["date"].dt.weekday

    df["attendance_ratio"] = df["actual_people"] / df["expected_people"].replace(0, np.nan)
    df["waste_percentage"] = df["leftover_kg"] / df["quantity_prepared_kg"].replace(0, np.nan)
    df["attendance_ratio"] = df["attendance_ratio"].fillna(0)
    df["waste_percentage"] = df["waste_percentage"].fillna(0)

    scaler = StandardScaler()
    scale_columns = [col for col in SCALE_COLUMNS if col in df.columns]
    scaler.fit(df[scale_columns])
    return scaler


if USE_S3:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is required when USE_S3=true.")
    MODEL = _load_model_from_s3(S3_BUCKET, S3_MODEL_KEY)
else:
    MODEL = joblib.load(MODEL_PATH)
TRAINING_COLUMNS = load_training_columns()
SCALER = build_scaler()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form.to_dict()
        features = build_feature_frame(form_data)

        for col in TRAINING_COLUMNS:
            if col not in features.columns:
                features[col] = 0

        features = features[TRAINING_COLUMNS]

        scale_cols = [col for col in SCALE_COLUMNS if col in features.columns]
        features[scale_cols] = SCALER.transform(features[scale_cols])

        prediction = float(MODEL.predict(features)[0])

        prepared = float(form_data["quantity_prepared_kg"])
        ratio = prediction / prepared if prepared > 0 else 0
        if ratio < 0.1:
            risk = "Low"
            recommendation = "Keep current preparation plan and monitor attendance."
        elif ratio < 0.25:
            risk = "Medium"
            recommendation = "Slightly reduce preparation or improve forecasting for this meal."
        else:
            risk = "High"
            recommendation = "Reduce preparation and increase redistribution planning."

        if USE_S3 and S3_BUCKET:
            log_payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input": form_data,
                "predicted_leftover_kg": round(prediction, 2),
                "waste_risk_level": risk,
                "recommendation": recommendation,
            }
            _upload_prediction_log(S3_BUCKET, log_payload)

        return render_template(
            "result.html",
            prediction=round(prediction, 2),
            risk=risk,
            recommendation=recommendation,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
