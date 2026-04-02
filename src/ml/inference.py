import joblib


def load_model(model_path: str):
    return joblib.load(model_path)


def run_inference(model, payload: dict):
    raise NotImplementedError("Implement prediction logic.")
