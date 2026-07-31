import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from project_utils import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_TRACKING_URI,
    INTERIM_DATA_DIR,
    PROJECT_ROOT,
    get_env,
    setup_logger,
)
from src.data.data_ingestion import load_params

logger = setup_logger("model_evaluation", "model_evaluation_errors.log")


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = PROJECT_ROOT / f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(str(cm_file_path))
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    tracking_uri = get_env("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    experiment_name = get_env("MLFLOW_EXPERIMENT", DEFAULT_MLFLOW_EXPERIMENT)
    root_dir = PROJECT_ROOT

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        try:
            params = load_params(str(root_dir / "params.yaml"))

            for key, value in params.items():
                mlflow.log_param(key, json.dumps(value) if isinstance(value, dict) else value)

            model = load_model(str(root_dir / "lgbm_model.pkl"))
            vectorizer = load_vectorizer(str(root_dir / "tfidf_vectorizer.pkl"))
            test_data = load_data(str(INTERIM_DATA_DIR / "test_processed.csv"))

            X_test_tfidf = vectorizer.transform(test_data["clean_comment"].values)
            y_test = test_data["category"].values

            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5],
                columns=vectorizer.get_feature_names_out(),
            )
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example,
            )

            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, str(root_dir / "experiment_info.json"))

            mlflow.log_artifact(str(root_dir / "tfidf_vectorizer.pkl"))

            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics["precision"],
                        f"test_{label}_recall": metrics["recall"],
                        f"test_{label}_f1-score": metrics["f1-score"],
                    })

            log_confusion_matrix(cm, "Test Data")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.exception("Failed to complete model evaluation: %s", e)
            raise

if __name__ == '__main__':
    main()
