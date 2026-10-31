from pathlib import Path

import pandas as pd
import yaml

from project_utils import DEFAULT_DATASET_URL, PROJECT_ROOT, RAW_DATA_DIR, get_env, setup_logger
from src.data.contracts import LABEL_COLUMN, label_distribution, stratified_split, validate_dataset

logger = setup_logger("data_ingestion", "data_ingestion_errors.log")

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        return validate_dataset(df)
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: Path,
    *,
    random_state: int,
    test_size: float,
) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(data_path / "train.csv", index=False)
        test_data.to_csv(data_path / "test.csv", index=False)
        metadata = {
            "text_column": "clean_comment",
            "label_column": LABEL_COLUMN,
            "labels": [-1, 0, 1],
            "random_state": random_state,
            "test_size": test_size,
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "train_distribution": label_distribution(train_data),
            "test_distribution": label_distribution(test_data),
        }
        (data_path / "split_metadata.json").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )
        logger.info("Train and test data saved to %s", data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path=str(PROJECT_ROOT / "params.yaml"))
        test_size = params["data_ingestion"]["test_size"]
        random_state = params["data_ingestion"].get("random_state", 42)
        data_url = get_env("DATASET_URL", DEFAULT_DATASET_URL)

        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)

        train_data, test_data = stratified_split(
            final_df, test_size=test_size, random_state=random_state
        )

        save_data(
            train_data,
            test_data,
            data_path=RAW_DATA_DIR,
            random_state=random_state,
            test_size=test_size,
        )
    except Exception as e:
        logger.exception("Failed to complete the data ingestion process: %s", e)
        raise

if __name__ == '__main__':
    main()
