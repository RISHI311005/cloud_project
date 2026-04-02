import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    port: int = int(os.getenv("PORT", "8000"))
    model_path: str = os.getenv("MODEL_PATH", "models/trained/model.joblib")
    cloud_provider: str = os.getenv("CLOUD_PROVIDER", "aws")


settings = Settings()
