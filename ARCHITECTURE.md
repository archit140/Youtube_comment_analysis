# Architecture

## High-Level Structure

The project is organized as an ML pipeline plus an inference service.

- `src/` contains the actual data and model pipeline.
- `flask_app/` contains the runtime inference API.
- `project_utils.py` holds shared configuration, preprocessing, paths, and logger setup.
- `dvc.yaml` describes the order of pipeline stages and their dependencies.
- `params.yaml` holds model hyperparameters used by the training stage.

## Main Components

### 1. Data Ingestion

File: `src/data/data_ingestion.py`

Responsibilities:

- Load pipeline parameters from `params.yaml`
- Download the labeled dataset from a remote CSV URL
- Drop nulls, duplicates, and empty comments
- Split the dataset into train and test sets
- Save the split files into `data/raw/`

Why it matters:

- This stage establishes the reproducible starting point for the entire project.
- The train/test split is fixed with `random_state=42`, which is an interview-worthy reproducibility choice.

### 2. Data Preprocessing

File: `src/data/data_preprocessing.py`

Responsibilities:

- Normalize text
- lowercase conversion
- whitespace cleanup
- newline removal
- special-character cleanup
- stopword removal with preserved negation words
- lemmatization
- Save processed outputs into `data/interim/`

Why it matters:

- This is the bridge between raw text and ML-ready features.
- Preserving words like `not` and `no` is a domain-aware sentiment decision.
- The shared preprocessing now lives in `project_utils.py` so training and inference use the same core logic.

### 3. Model Training and Evaluation

Files: `src/models/` and `src/training/`

Responsibilities:

- Build shared TF-IDF pipelines for four traditional models.
- Train the CPU-friendly PyTorch MLP baseline.
- Use a stratified validation split while keeping the test set untouched.
- Compute accuracy, macro/weighted metrics, per-class metrics, timings, and confusion matrices.
- Save local model artifacts and measured comparison reports.

Important design decisions:

- `ngram_range: [1, 3]` provides unigram, bigram, and trigram features.
- Macro F1 is used for model comparison because the sentiment classes are imbalanced.
- The full preprocessing/model pipeline is saved for consistent inference.

### 5. Local Artifact Selection

The primary serving artifact is selected with `LOCAL_MODEL_PATH` and defaults to
`models/lightgbm.joblib`. MLflow remains available for local experiment tracking
and an explicitly configured local `MLFLOW_MODEL_URI`, but the API does not
depend on a registry or remote service.

### 5. Flask Inference Service

File: `flask_app/app.py`

Responsibilities:

- Load the local serialized model pipeline, with an optional local MLflow URI fallback
- Apply inference-time preprocessing
- Predict sentiment for comment lists
- Generate chart, word cloud, and trend graph responses
- Retry loading artifacts on request instead of crashing the server at startup

Endpoints:

- `GET /`
- `POST /predict`
- `POST /predict_with_timestamps`
- `POST /generate_chart`
- `POST /generate_wordcloud`
- `POST /generate_trend_graph`

## End-To-End Request / Data Flow

### Training Flow

1. `data_ingestion.py` explicitly downloads and splits the dataset.
2. `data_preprocessing.py` normalizes raw comments.
3. `src/training/train.py` fits the selected TF-IDF/classifier pipeline.
4. `src/training/compare.py` evaluates all selected models on the untouched test set.
5. The selected local artifact is loaded directly by Flask and Streamlit.

### Inference Flow

1. Client sends comments to the Flask API.
2. API preprocesses each comment.
3. The saved pipeline transforms comments and predicts class labels.
4. Confidence is returned when the classifier supports probabilities.
5. API returns sentiment labels or derived visualizations.


