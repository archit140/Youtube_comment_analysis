import pandas as pd
from project_utils import INTERIM_DATA_DIR, RAW_DATA_DIR, preprocess_comment, setup_logger

logger = setup_logger("data_preprocessing", "data_preprocessing_errors.log")

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(data_path / "train_processed.csv", index=False)
        test_data.to_csv(data_path / "test_processed.csv", index=False)
        logger.info("Processed data saved to %s", data_path)
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.info("Starting data preprocessing...")
        train_data = pd.read_csv(RAW_DATA_DIR / "train.csv")
        test_data = pd.read_csv(RAW_DATA_DIR / "test.csv")
        logger.info("Data loaded successfully")

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        save_data(train_processed_data, test_processed_data, data_path=INTERIM_DATA_DIR)
    except Exception as e:
        logger.exception("Failed to complete the data preprocessing process: %s", e)
        raise

if __name__ == '__main__':
    main()
