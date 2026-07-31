from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from project_utils import DEFAULT_DATASET_URL, PROJECT_ROOT, RAW_DATA_DIR, get_env, setup_logger

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
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']
        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: Path) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(data_path / "train.csv", index=False)
        test_data.to_csv(data_path / "test.csv", index=False)
        logger.info("Train and test data saved to %s", data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path=str(PROJECT_ROOT / "params.yaml"))
        test_size = params["data_ingestion"]["test_size"]
        data_url = get_env("DATASET_URL", DEFAULT_DATASET_URL)

        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        save_data(train_data, test_data, data_path=RAW_DATA_DIR)
    except Exception as e:
        logger.exception("Failed to complete the data ingestion process: %s", e)
        raise

if __name__ == '__main__':
    main()
