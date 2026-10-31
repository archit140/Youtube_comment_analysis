"""Traditional text-classification models with a shared interface."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def available_traditional_models() -> tuple[str, ...]:
    return ("logistic_regression", "naive_bayes", "linear_svm", "lightgbm")


def available_models() -> tuple[str, ...]:
    return (*available_traditional_models(), "mlp")


def build_model(
    name: str,
    config: dict | None = None,
    *,
    random_state: int = 42,
    max_features: int = 10001,
    ngram_range: tuple[int, int] = (1, 3),
) -> Pipeline:
    """Build a TF-IDF pipeline for a supported model name."""
    config = config or {}
    random_state = config.get("random_state", random_state)
    max_features = config.get("max_features", max_features)
    ngram_range = tuple(config.get("ngram_range", ngram_range))
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    if name == "logistic_regression":
        classifier: Any = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        )
    elif name == "naive_bayes":
        classifier = MultinomialNB()
    elif name == "linear_svm":
        classifier = LinearSVC(class_weight="balanced", random_state=random_state)
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is not installed; choose another model.") from exc
        classifier = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            learning_rate=config.get("learning_rate", 0.09),
            max_depth=config.get("max_depth", 20),
            n_estimators=config.get("n_estimators", 367),
            random_state=random_state,
            verbosity=-1,
        )
    else:
        choices = ", ".join(available_traditional_models())
        raise ValueError(f"Unsupported model '{name}'. Choose one of: {choices}")

    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
