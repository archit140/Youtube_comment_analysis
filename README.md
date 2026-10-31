# YouTube Comment Sentiment Analysis

This project predicts sentiment for YouTube comments and exposes the trained model through a Flask API and Streamlit dashboard. It is organized as a local ML workflow with preprocessing, model comparison, experiment tracking, API serving, and reproducible pipeline configuration.

## What The Project Does

- Cleans and normalizes text comments with NLTK-based preprocessing
- Splits raw labeled data into train and test sets
- Builds TF-IDF text features with unigram to trigram support
- Trains Naive Bayes, Logistic Regression, Linear SVM, LightGBM, and a CPU-friendly PyTorch MLP
- Evaluates the model and logs metrics/artifacts to MLflow
- Serves local predictions and visualization endpoints through Flask

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
|   |-- models/
|   |   |-- traditional.py      # Shared traditional model factory
|   |   `-- mlp.py              # Optional PyTorch MLP baseline
|   |-- training/
|   |   |-- train.py            # Unified model training CLI
|   |   `-- compare.py          # Measured model comparison
|-- dashboard/app.py            # Local Streamlit dashboard
|-- dvc.yaml                    # Pipeline orchestration
|-- dvc.lock                    # Pinned pipeline outputs
|-- params.yaml                 # Training hyperparameters
|-- tests/                      # API and preprocessing tests
|-- Procfile                    # Local Waitress entrypoint
|-- Dockerfile                  # Optional local container
`-- AGENTS.md                   # Contributor and agent guidelines
```

## Architecture Overview

The project has two main layers:

1. Offline ML pipeline
   - `src/data/data_ingestion.py` downloads the labeled dataset and writes raw train/test CSV files.
   - `src/data/data_preprocessing.py` cleans the text and saves processed datasets.
   - `src/training/train.py` trains the selected traditional model or MLP.
   - `src/training/compare.py` evaluates all selected models, including the MLP, and writes measured reports.
   - MLflow stores local experiment metadata and model artifacts.

2. Online inference layer
   - `flask_app/app.py` loads a local pipeline or local MLflow model and exposes prediction endpoints.
   - `dashboard/app.py` provides the local interactive demonstration interface.

More detail is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Data Flow

1. Raw dataset is fetched from the external CSV URL in `data_ingestion.py`.
2. The dataset is cleaned and split into `data/raw/train.csv` and `data/raw/test.csv`.
3. Comments are normalized in `data_preprocessing.py` and saved to `data/interim/`.
4. `src/training/train.py` fits TF-IDF and the selected classifier using only the training split.
5. The complete pipeline is serialized under `models/` and logged to local MLflow.
6. `src/training/compare.py` writes measured metrics, timings, model sizes, and confusion matrices.
7. `flask_app/app.py` loads the local pipeline and returns predictions.
8. `dashboard/app.py` provides the local interactive demonstration interface.

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

You can run the data stages directly:

```bash
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
```

Or use DVC:

```bash
dvc repro
```

To train a selected traditional model directly:

```bash
python -m src.training.train --model logistic_regression
python -m src.training.train --model naive_bayes
python -m src.training.train --model linear_svm
python -m src.training.train --model lightgbm
```

Each run uses a stratified validation split, keeps the test set untouched, logs
metrics to the local MLflow database, and writes ignored artifacts under `models/`.

Compare all supported models:

```bash
python -m src.training.compare --models naive_bayes logistic_regression linear_svm lightgbm mlp
```

The measured comparison is written to `reports/model_comparison.csv` and
`reports/model_comparison.json`. It includes test metrics, per-class metrics,
training/inference timing, artifact size, and probability support.

Current measured test comparison (generated locally):

| Model | Accuracy | Macro F1 | Weighted F1 | Training (s) | Inference (s) | Probability |
|---|---:|---:|---:|---:|---:|---|
| LightGBM | 0.9038 | 0.8945 | 0.9028 | 58.74 | 0.81 | Yes |
| Linear SVM | 0.8824 | 0.8717 | 0.8813 | 7.11 | 0.65 | No |
| MLP | 0.8621 | 0.8475 | 0.8602 | 26.92 | 0.80 | Yes |
| Logistic Regression | 0.8473 | 0.8376 | 0.8468 | 7.71 | 0.68 | Yes |
| Naive Bayes | 0.6313 | 0.5686 | 0.6006 | 6.10 | 0.55 | Yes |

## Running The Flask API

```bash
python flask_app/app.py
```

Default base URL:

```text
http://127.0.0.1:5000
```

## Running the local dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard supports single-comment predictions, CSV batch inference,
downloadable results, sentiment counts, and measured model comparisons.

To analyze a YouTube video interactively, create a Google/YouTube Data API key,
place it in the local `.env` file as `YOUTUBE_API_KEY`, restart Streamlit, and
paste a video URL into the **YouTube Analysis** tab. The app retrieves top-level
comments, runs the saved local model, and shows positive/neutral/negative counts,
percentages, a chart, a word cloud, and downloadable predictions. Without an API
key, the manual and CSV workflows remain available.

To inspect local MLflow runs:

```bash
mlflow ui --backend-store-uri sqlite:///./mlflow.db --host 127.0.0.1 --port 5001 --workers 1
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
| `MLFLOW_TRACKING_URI` | Local MLflow SQLite tracking database | `sqlite:///./mlflow.db` |
| `MLFLOW_EXPERIMENT` | MLflow experiment name | `yt-comment-analysis` |
| `MLFLOW_MODEL_URI` | Optional explicit local MLflow model URI fallback | unset |
| `LOCAL_MODEL_PATH` | Primary locally saved model pipeline | `./models/lightgbm.joblib` |
| `YOUTUBE_API_KEY` | Optional key for explicit YouTube comment retrieval | unset |
| `PORT` | Flask server port | `5000` |
| `FLASK_DEBUG` | Enables Flask debug mode locally | `false` |

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## Local serving and containerization

### Option 1: Local Waitress server

The repository includes a [Procfile](Procfile) that serves the app with Waitress:

```text
web: waitress-serve --listen=0.0.0.0:$PORT wsgi:app
```

This command is intended for local serving only; it does not deploy or publish the application.

### Option 2: Local Docker container

Build:

```bash
docker build -t yt-comment-api .
```

Run:

```bash
docker run -p 8000:8000 yt-comment-api
```

### Local-only notes

- The API loads `LOCAL_MODEL_PATH` first; predictions require a locally trained artifact.
- MLflow stores metadata in `mlflow.db` and artifacts under the local project directory.
- DVC uses `.dvc/storage`; no DVC remote backend is configured.
- No cloud deployment, cloud credentials, or hosted inference service is required.

## Interview Notes

This project is especially useful for revising:

- NLP preprocessing
- TF-IDF feature engineering
- LightGBM for text classification
- experiment tracking with MLflow
- data pipelines with DVC
- model serving with Flask

See [INTERVIEW_NOTES.md](INTERVIEW_NOTES.md) for a focused revision guide.

## Important Context

- The training dataset source referenced in the pipeline is a labeled sentiment CSV hosted on GitHub. That means the project demonstrates transfer of a trained sentiment model to YouTube comment inference rather than building a labeled YouTube dataset from scratch.

## Future Improvements

- Add a small script for exporting a specific MLflow model version into fully local artifacts
- Add CI to run tests automatically on every push
- Add a simple frontend or sample client for easier demoing

