from ml.inference import load_model, run_inference
from config.settings import settings

_model = None


def get_prediction(payload: dict):
    global _model
    if _model is None:
        _model = load_model(settings.model_path)
    return run_inference(_model, payload)
