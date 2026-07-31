# YouTube Comment Sentiment Analysis

This project predicts sentiment for YouTube comments and exposes the trained model through a small Flask API. It is now organized as a focused end-to-end ML workflow with preprocessing, training, experiment tracking, model registration, API serving, and deployment-ready project structure.

## What The Project Does

- Cleans and normalizes text comments with NLTK-based preprocessing
- Splits raw labeled data into train and test sets
- Builds TF-IDF text features with unigram to trigram support
- Trains a LightGBM multiclass sentiment classifier
- Evaluates the model and logs metrics/artifacts to MLflow
- Registers the trained model in MLflow Model Registry
- Serves prediction and visualization endpoints through Flask

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- LightGBM
- NLTK
- MLflow
- DVC
- Flask, flask-cors
- matplotlib, seaborn, wordcloud

## Repository Structure

```text
yt_comment/
|-- flask_app/
|   `-- app.py                  # Inference API and visualization endpoints
|-- src/
|   |-- data/
|   |   |-- data_ingestion.py   # Download, clean, split, save raw train/test data
|   |   `-- data_preprocessing.py
|   `-- model/
|       |-- model_building.py   # TF-IDF + LightGBM training
|       |-- model_evaluation.py # MLflow logging, confusion matrix, metrics
|       `-- register_model.py   # Register trained model in MLflow
|-- notebooks/                  # Experiments and iteration history
|-- dvc.yaml                    # Pipeline orchestration
|-- dvc.lock                    # Pinned pipeline outputs
|-- params.yaml                 # Training hyperparameters
|-- tests/                      # API and preprocessing tests
|-- confusion_matrix_Test Data.png
|-- Procfile                    # Simple platform deployment entrypoint
|-- Dockerfile                  # Container deployment option
|-- tfidf_vectorizer.pkl        # Generated artifact after training
`-- lgbm_model.pkl              # Generated artifact after training
```

## Architecture Overview

The project has two main layers:

1. Offline ML pipeline
   - `src/data/data_ingestion.py` downloads the labeled dataset and writes raw train/test CSV files.
   - `src/data/data_preprocessing.py` cleans the text and saves processed datasets.
   - `src/model/model_building.py` creates TF-IDF features and trains LightGBM.
   - `src/model/model_evaluation.py` evaluates the model and logs the run to MLflow.
   - `src/model/register_model.py` registers the trained model in MLflow for serving.

2. Online inference layer
   - `flask_app/app.py` loads the registered model plus the serialized vectorizer and exposes API endpoints for predictions and visual summaries.

More detail is in [ARCHITECTURE.md](C:\Users\archi\OneDrive\Desktop\yt_comment\ARCHITECTURE.md).

## Data Flow

1. Raw dataset is fetched from the external CSV URL in `data_ingestion.py`.
2. The dataset is cleaned and split into `data/raw/train.csv` and `data/raw/test.csv`.
3. Comments are normalized in `data_preprocessing.py` and saved to `data/interim/`.
4. `model_building.py` fits TF-IDF and LightGBM using the training split.
5. The trained model and vectorizer are serialized.
6. `model_evaluation.py` logs metrics, confusion matrix, and model metadata to MLflow.
7. `register_model.py` registers the trained model for inference.
8. `flask_app/app.py` receives comments, applies the same preprocessing, vectorizes them, and returns predicted sentiment.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template if you want to override defaults:

```bash
copy .env.example .env
```

## Running The Pipeline

You can run the stages directly:

```bash
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

Or use DVC:

```bash
dvc repro
```

## Running The Flask API

```bash
python flask_app/app.py
```

Default base URL:

```text
http://127.0.0.1:5000
```

## Example API Usage

Predict plain comments:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"comments\": [\"This video was amazing\", \"I did not like this at all\"]}"
```

Predict comments with timestamps:

```bash
curl -X POST http://127.0.0.1:5000/predict_with_timestamps ^
  -H "Content-Type: application/json" ^
  -d "{\"comments\": [{\"text\": \"Great explanation\", \"timestamp\": \"2025-12-01T10:00:00\"}]}"
```

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `DATASET_URL` | Remote CSV used during data ingestion | Reddit sentiment CSV used in the original project |
| `MLFLOW_TRACKING_URI` | MLflow tracking server used by training and the Flask app | `http://127.0.0.1:5000` |
| `MLFLOW_EXPERIMENT` | MLflow experiment name | `yt-comment-analysis` |
| `MLFLOW_MODEL_NAME` | Registered model name | `yt_chrome_plugin_model` |
| `MLFLOW_MODEL_VERSION` | Registered model version | `1` |
| `VECTORIZER_PATH` | Local TF-IDF vectorizer path | `./tfidf_vectorizer.pkl` |
| `PORT` | Flask server port | `5000` |
| `FLASK_DEBUG` | Enables Flask debug mode locally | `false` |

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## Deployment

### Option 1: Procfile-style deployment

The repository includes a [Procfile](C:\Users\archi\OneDrive\Desktop\yt_comment\Procfile:1) that serves the app with Waitress:

```text
web: waitress-serve --listen=0.0.0.0:$PORT wsgi:app
```

This is suitable for platforms that support Procfile-based Python apps.

### Option 2: Docker deployment

Build:

```bash
docker build -t yt-comment-api .
```

Run:

```bash
docker run -p 8000:8000 yt-comment-api
```

### Deployment Notes

- The API can start without the MLflow model being immediately available, but predictions require a reachable MLflow server and the saved vectorizer file.
- If you want a fully self-contained deployment, keep `lgbm_model.pkl` and `tfidf_vectorizer.pkl` with the app or adapt the loader to use only local artifacts.
- Before public deployment, set the environment variables for your own MLflow server and model version.


## Important Context

- The training dataset source referenced in the pipeline is a labeled sentiment CSV hosted on GitHub. That means the project demonstrates transfer of a trained sentiment model to YouTube comment inference rather than building a labeled YouTube dataset from scratch.

## Future Improvements

- Add a small script for exporting a specific MLflow model version into fully local artifacts
- Add CI to run tests automatically on every push
- Add a simple frontend or sample client for easier demoing

