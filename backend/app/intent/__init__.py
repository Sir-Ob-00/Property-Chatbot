from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[3] / "artifacts/intent_classifier.joblib"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Train the intent classifier and restart the API.")
    return joblib.load(MODEL_PATH)


def classify(message: str) -> tuple[str, float]:
    model = load_model()
    probabilities = model.predict_proba([message])[0]
    best = probabilities.argmax()
    return str(model.classes_[best]), float(probabilities[best])
