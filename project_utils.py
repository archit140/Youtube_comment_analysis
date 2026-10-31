import logging
import os
import re
from pathlib import Path

import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/Himanshu-1703/"
    "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
)

DEFAULT_MLFLOW_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve().as_posix()}"
)
DEFAULT_MLFLOW_EXPERIMENT = "yt-comment-analysis"
DEFAULT_MODEL_NAME = "local_lightgbm"


def get_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def setup_logger(name: str, error_log_name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOGS_DIR / error_log_name)
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        if get_env("ALLOW_NLTK_DOWNLOAD", "false").lower() != "true":
            raise RuntimeError(
                f"Missing NLTK resource '{package_name}'. Install it before offline execution "
                "or set ALLOW_NLTK_DOWNLOAD=true for an explicit one-time download."
            )
        nltk.download(package_name, quiet=True)


def ensure_nltk_dependencies() -> None:
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    ensure_nltk_resource("corpora/wordnet", "wordnet")
    ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")


def preprocess_comment(comment) -> str:
    if comment is None:
        return ""

    try:
        ensure_nltk_dependencies()
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
    except (LookupError, RuntimeError):
        # Keep inference usable offline when optional NLTK corpora have not been
        # provisioned yet. The training environment should install the corpora
        # for the full stopword-removal and lemmatization path.
        stop_words = set()
        lemmatizer = None

    comment = str(comment).lower().strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    retained_words = {"not", "but", "however", "no", "yet"}
    stop_words -= retained_words
    comment = " ".join(word for word in comment.split() if word not in stop_words)

    if lemmatizer is None:
        return comment
    return " ".join(lemmatizer.lemmatize(word) for word in comment.split())
