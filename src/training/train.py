"""Train one supported model using the local processed dataset."""

from __future__ import annotations

import argparse
import json
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from project_utils import DEFAULT_MLFLOW_EXPERIMENT, INTERIM_DATA_DIR, PROJECT_ROOT
from src.models import available_models, build_model
from src.models.mlp import TextMlpClassifier
from src.training.evaluation import evaluate_classifier, write_json


def _load_data():
    train_path = INTERIM_DATA_DIR / "train_processed.csv"
    test_path = INTERIM_DATA_DIR / "test_processed.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed data is missing. Run the explicit ingestion and preprocessing commands first."
        )
    train_df = pd.read_csv(train_path).dropna(subset=["clean_comment", "category"])
    test_df = pd.read_csv(test_path).dropna(subset=["clean_comment", "category"])
    return train_df, test_df


def train(model_name: str, *, random_state: int = 42) -> dict:
    train_df, test_df = _load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_df["clean_comment"],
        train_df["category"],
        test_size=0.2,
        random_state=random_state,
        stratify=train_df["category"],
    )

    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
    model_config = params.get("model_building", {})
    if model_name == "mlp":
        mlp_config = params.get("mlp", {})
        model = TextMlpClassifier(**mlp_config, random_state=random_state)
    else:
        model = build_model(model_name, model_config, random_state=random_state)
    started = time.perf_counter()
    if model_name == "mlp":
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid))
    else:
        model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    validation = evaluate_classifier(model, x_valid, y_valid)
    test = evaluate_classifier(model, test_df["clean_comment"], test_df["category"])

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{model_name}.joblib"
    if model_name == "mlp":
        artifact_path = model_dir / "mlp.pt"
        model.save(artifact_path)
    else:
        joblib.dump(model, artifact_path)

    result = {
        "model_name": model_name,
        "random_state": random_state,
        "training_seconds": training_seconds,
        "validation": validation,
        "test": test,
        "supports_probability": test["supports_probability"],
        "artifact_path": str(artifact_path),
        "model_size_bytes": artifact_path.stat().st_size,
    }
    report_path = model_dir / f"{model_name}_report.json"
    write_json(report_path, result)

    tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DEFAULT_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({"model": model_name, "random_state": random_state})
        mlflow.log_metrics({
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"],
            "validation_accuracy": validation["accuracy"],
            "training_seconds": training_seconds,
            "inference_seconds": test["inference_seconds"],
            "model_size_bytes": result["model_size_bytes"],
        })
        mlflow.set_tag("supports_probability", str(result["supports_probability"]))
        mlflow.log_artifact(str(artifact_path))
        mlflow.log_artifact(str(report_path))
        if model_name != "mlp":
            try:
                mlflow.sklearn.log_model(model, "model")
                result["model_uri"] = f"runs:/{run.info.run_id}/model"
            except (OSError, PermissionError) as exc:
                # The local joblib artifact remains the serving source of truth.
                # Some locked-down Windows environments deny MLflow's temporary
                # model directory while still allowing normal artifact logging.
                mlflow.set_tag("sklearn_model_flavor_status", "skipped")
                mlflow.set_tag("sklearn_model_flavor_error", type(exc).__name__)
        result["run_id"] = run.info.run_id
    write_json(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=available_models(), required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.model, random_state=args.random_state), indent=2))


if __name__ == "__main__":
    main()
