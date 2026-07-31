import json
from pathlib import Path

import mlflow

from project_utils import (
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME,
    PROJECT_ROOT,
    get_env,
    setup_logger,
)

logger = setup_logger("model_registration", "model_registration_errors.log")

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        mlflow.set_tracking_uri(get_env("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI))
        model_info = load_model_info(str(PROJECT_ROOT / "experiment_info.json"))
        model_name = get_env("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
        register_model(model_name, model_info)
    except Exception as e:
        logger.exception("Failed to complete the model registration process: %s", e)
        raise

if __name__ == '__main__':
    main()
