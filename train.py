from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import load_corpus

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train_model(test_size: float = 0.2, random_state: int = 42) -> dict:
    texts, labels = load_corpus(DATA_DIR)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Apache SpamAssassin Public Corpus (easy_ham and spam)",
        "total_messages": len(texts),
        "train_messages": len(x_train),
        "test_messages": len(x_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": ["ham", "spam"],
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=["ham", "spam"], output_dict=True, zero_division=0
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_model()
    print(f"Training complete. Accuracy: {result['accuracy']:.3f}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
