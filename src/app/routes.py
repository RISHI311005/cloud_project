from flask import Blueprint, jsonify, request

from app.services.prediction_service import get_prediction

api = Blueprint("api", __name__)


@api.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@api.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    prediction = get_prediction(payload)
    return jsonify({"prediction": prediction})
