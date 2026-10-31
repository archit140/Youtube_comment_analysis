"""Shared classification evaluation helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(model, x_test, y_test) -> dict:
    started = time.perf_counter()
    predictions = model.predict(x_test)
    inference_seconds = time.perf_counter() - started
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x_test)
        except (AttributeError, NotImplementedError, ValueError):
            probabilities = None
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[-1, 0, 1]).tolist(),
        "inference_seconds": inference_seconds,
        "supports_probability": probabilities is not None,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
