import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def train():
    with (ROOT / "training/data/intents.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    messages = [row["text"].strip() for row in rows]
    labels = [row["intent"].strip() for row in rows]
    if not rows or not all(messages) or not all(labels):
        raise ValueError("Every dataset row needs a message and an intent.")
    if len(set(message.casefold() for message in messages)) != len(messages):
        raise ValueError("Remove duplicate messages before training.")
    x_train, x_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.25, random_state=42, stratify=labels,
    )
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0),
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    print(classification_report(y_test, predictions, zero_division=0))
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(pipeline, artifacts / "intent_classifier.joblib")
    metadata = {
        "scikit_learn_version": sklearn.__version__,
        "seed": 42,
        "label_counts": dict(Counter(labels)),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "evaluation": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }
    (artifacts / "intent_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train()
