"""Training script for food waste prediction models."""

from __future__ import annotations

import os
from io import StringIO
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def load_dataset(path: str) -> pd.DataFrame:
    """Load processed dataset from CSV (local or S3)."""
    use_s3 = os.getenv("USE_S3", "false").lower() == "true"
    if not use_s3:
        return pd.read_csv(path)

    import boto3

    bucket = os.environ["S3_BUCKET"]
    processed_key = os.getenv("S3_PROCESSED_KEY", "processed/cleaned_data.csv")
    region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    s3 = boto3.client("s3", region_name=region)
    obj = s3.get_object(Bucket=bucket, Key=processed_key)
    return pd.read_csv(obj["Body"])


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
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


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print evaluation metrics in a readable format."""
    print("Model Performance (lower MAE/RMSE is better, higher R2 is better)")
    for name, metrics in results.items():
        print(
            f"- {name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}"
        )


def show_feature_importance(model, feature_names) -> None:
    """Print feature importance for the random forest model."""
    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(10)
    print("\nTop 10 Random Forest Feature Importances:")
    for name, value in importance.items():
        print(f"- {name}: {value:.4f}")


def main() -> None:
    """Train, evaluate, and save the best model."""
    input_path = "data/processed/cleaned_data.csv"
    output_path = os.path.join("model", "model.pkl")

    df = load_dataset(input_path)
    X, y = split_features_target(df, target="leftover_kg")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}

    print_results(results)
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    if best_model_name == "RandomForest":
        show_feature_importance(best_model, X.columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"\nSaved best model to {output_path}")


if __name__ == "__main__":
    main()
