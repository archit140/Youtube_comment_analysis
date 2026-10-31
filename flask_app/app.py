"""
Flask API for serving the YouTube comment sentiment model.

This keeps the original endpoints but makes local usage easier by:
- loading model settings from environment variables
- starting even when the model is temporarily unavailable
- returning clearer JSON errors for invalid payloads
"""

import io
import logging
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import stopwords
from scipy import sparse

from project_utils import (
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME,
    PROJECT_ROOT,
    preprocess_comment,
)

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


_MODEL = None
_VECTORIZER = None
_MODEL_LOADING = False


def load_model_and_vectorizer(
    model_name, model_version, vectorizer_path, tracking_uri=None, local_model_path=None
):
    """Load a local pipeline first, then fall back to the local MLflow registry."""
    if local_model_path:
        local_model_path = os.path.abspath(
            local_model_path if os.path.isabs(local_model_path) else PROJECT_ROOT / local_model_path
        )
    if local_model_path and os.path.exists(local_model_path):
        logger.info("Loading local pipeline from %s", local_model_path)
        if local_model_path.lower().endswith(".pt"):
            from src.models.mlp import TextMlpClassifier

            return TextMlpClassifier.load(local_model_path), None
        return joblib.load(local_model_path), None

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.environ.get("MLFLOW_MODEL_URI")
    if not model_uri:
        raise RuntimeError(
            "No local model found. Set LOCAL_MODEL_PATH to a trained model artifact."
        )
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def get_model_vectorizer():
    """Return cached artifacts or load them on demand."""
    global _MODEL, _VECTORIZER, _MODEL_LOADING

    if _MODEL is not None:
        return _MODEL, _VECTORIZER

    if _MODEL_LOADING:
        time.sleep(0.5)
        if _MODEL is not None and _VECTORIZER is not None:
            return _MODEL, _VECTORIZER

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )
    vectorizer_path = os.environ.get("VECTORIZER_PATH", "./tfidf_vectorizer.pkl")
    local_model_path = os.environ.get(
        "LOCAL_MODEL_PATH", str(PROJECT_ROOT / "models" / "lightgbm.joblib")
    )

    try:
        _MODEL_LOADING = True
        _MODEL, _VECTORIZER = load_model_and_vectorizer(
            model_name=DEFAULT_MODEL_NAME,
            model_version="1",
            vectorizer_path=vectorizer_path,
            tracking_uri=tracking_uri,
            local_model_path=local_model_path,
        )
        logger.info("Model and vectorizer loaded successfully.")
        return _MODEL, _VECTORIZER
    finally:
        _MODEL_LOADING = False


def predict_from_vectorizer(model, vectorizer, preprocessed_comments):
    """Transform text with TF-IDF and make predictions using the loaded model."""
    predictions, _ = predict_outputs(model, vectorizer, preprocessed_comments)
    return predictions


def predict_outputs(model, vectorizer, preprocessed_comments):
    """Return predictions and optional confidence scores."""
    if vectorizer is None:
        model_input = preprocessed_comments
    else:
        model_input = vectorizer.transform(preprocessed_comments)
        if not sparse.issparse(model_input):
            model_input = np.asarray(model_input)

    predictions = np.asarray(model.predict(model_input)).tolist()
    confidences = [None] * len(predictions)
    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(model_input))
            confidences = [round(float(value), 6) for value in probabilities.max(axis=1)]
        except (AttributeError, NotImplementedError, ValueError):
            pass
    return list(map(str, predictions)), confidences


def current_model_name(model=None) -> str:
    """Return a stable display name for the active local model."""
    configured = os.environ.get("LOCAL_MODEL_PATH")
    if configured:
        return os.path.splitext(os.path.basename(configured))[0]
    if model is not None and model.__class__.__name__:
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            classifier = model.named_steps["classifier"]
            classifier_names = {
                "LGBMClassifier": "lightgbm",
                "LogisticRegression": "logistic_regression",
                "LinearSVC": "linear_svm",
                "MultinomialNB": "naive_bayes",
            }
            return classifier_names.get(
                classifier.__class__.__name__, classifier.__class__.__name__.lower()
            )
        return model.__class__.__name__
    return DEFAULT_MODEL_NAME


try:
    get_model_vectorizer()
except Exception as exc:
    logger.warning(
        "Initial model load skipped (%s). The API will retry on prediction requests.",
        exc,
    )


@app.route("/")
def home():
    return "Welcome to the YouTube comment sentiment API."


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        comments_data = data.get("comments")
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        comments = [item.get("text") for item in comments_data]
        timestamps = [item.get("timestamp") for item in comments_data]
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        try:
            model, vectorizer = get_model_vectorizer()
        except Exception as exc:
            return jsonify({"error": "Model not loaded", "detail": str(exc)}), 503

        predictions, confidences = predict_outputs(model, vectorizer, preprocessed_comments)
        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "model": current_model_name(model),
                "timestamp": timestamp,
            }
            for comment, sentiment, confidence, timestamp in zip(
                comments, predictions, confidences, timestamps
            )
        ]
        return jsonify(response)
    except Exception as exc:
        logger.exception("Unhandled error in /predict_with_timestamps: %s", exc)
        return jsonify(
            {
                "error": "Internal server error",
                "detail": str(exc),
            }
        ), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        single_comment = data.get("comment")
        comments = data.get("comments")
        if single_comment is not None:
            comments = [single_comment]
            single_response = True
        else:
            single_response = False
        if not isinstance(comments, list) or not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        try:
            model, vectorizer = get_model_vectorizer()
        except Exception as exc:
            return jsonify({"error": "Model not loaded"}), 503

        predictions, confidences = predict_outputs(model, vectorizer, preprocessed_comments)
        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "model": current_model_name(model),
            }
            for comment, sentiment, confidence in zip(comments, predictions, confidences)
        ]
        return jsonify(response[0] if single_response else response)
    except Exception as exc:
        logger.exception("Unhandled error in /predict: %s", exc)
        return jsonify(
            {
                "error": "Internal server error",
            }
        ), 500


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """Predict a CSV containing a `comment`, `text`, or `clean_comment` column."""
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Upload a CSV file using the 'file' field"}), 400
    try:
        frame = pd.read_csv(uploaded)
        column = next((name for name in ("comment", "text", "clean_comment") if name in frame), None)
        if column is None:
            return jsonify({"error": "CSV must contain comment, text, or clean_comment"}), 400
        comments = frame[column].fillna("").astype(str).tolist()
        if not any(comment.strip() for comment in comments):
            return jsonify({"error": "CSV contains no non-empty comments"}), 400
        preprocessed = [preprocess_comment(comment) for comment in comments]
        model, vectorizer = get_model_vectorizer()
        predictions, confidences = predict_outputs(model, vectorizer, preprocessed)
        result = frame.copy()
        result["sentiment"] = predictions
        result["confidence"] = confidences
        output = io.BytesIO()
        result.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")
    except Exception as exc:
        logger.exception("CSV prediction failed: %s", exc)
        return jsonify({"error": "CSV prediction failed"}), 500


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        sentiment_counts = data.get("sentiment_counts")
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype="image/png")
    except Exception as exc:
        logger.exception("Error in /generate_chart: %s", exc)
        return jsonify({"error": f"Chart generation failed: {exc}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        from wordcloud import WordCloud

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        comments = data.get("comments")
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(" ".join(preprocessed_comments))

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as exc:
        logger.exception("Error in /generate_wordcloud: %s", exc)
        return jsonify({"error": f"Word cloud generation failed: {exc}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        sentiment_data = data.get("sentiment_data")
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=labels[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype="image/png")
    except Exception as exc:
        logger.exception("Error in /generate_trend_graph: %s", exc)
        return jsonify({"error": f"Trend graph generation failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
