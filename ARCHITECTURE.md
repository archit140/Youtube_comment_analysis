# Architecture

## High-Level Structure

The project is organized as an ML pipeline plus an inference service.

- `src/` contains the actual data and model pipeline.
- `flask_app/` contains the runtime inference API.
- `notebooks/` capture the experimentation path that led to the current model choices.
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

### 3. Model Building

File: `src/model/model_building.py`

Responsibilities:

- Load processed training data
- Convert comment text into TF-IDF features
- Configure and train a multiclass LightGBM classifier
- Serialize the vectorizer and trained model

Important design decisions:

- `ngram_range: [1, 3]` means the model uses unigram, bigram, and trigram information.
- `class_weight="balanced"` and `is_unbalance=True` reflect awareness of class imbalance.
- The vectorizer is saved separately because inference must reproduce the exact same feature space.

### 4. Model Evaluation

File: `src/model/model_evaluation.py`

Responsibilities:

- Load the saved model and vectorizer
- Transform the processed test dataset
- Compute classification metrics and confusion matrix
- Log parameters, metrics, model artifact, vectorizer, and confusion matrix to MLflow
- Save run metadata into `experiment_info.json`

Why it matters:

- This stage shows the difference between training a model and managing an experiment.
- It is the strongest MLOps-oriented part of the repo.

### 5. Model Registration

File: `src/model/register_model.py`

Responsibilities:

- Read the saved run metadata
- Register the model artifact from MLflow into the Model Registry
- Move the new model version to the `Staging` stage

Why it matters:

- This turns a local experiment into a deployable tracked model version.
- It shows understanding of the model lifecycle, not just notebook-level experimentation.

### 6. Flask Inference Service

File: `flask_app/app.py`

Responsibilities:

- Load MLflow model and local vectorizer
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

1. `data_ingestion.py` downloads and splits the dataset.
2. `data_preprocessing.py` normalizes raw comments.
3. `model_building.py` fits TF-IDF and trains LightGBM.
4. `model_evaluation.py` logs evaluation results to MLflow.
5. `register_model.py` registers the trained model version.

### Inference Flow

1. Client sends comments to the Flask API.
2. API preprocesses each comment.
3. Saved TF-IDF vectorizer transforms the comments.
4. Loaded MLflow model predicts class labels.
5. API returns sentiment labels or derived visualizations.

## Design Trade-Offs

- Strength:
  - The project separates offline training from online serving clearly.
- Strength:
  - DVC plus MLflow make the project more interview-ready than a notebook-only workflow.
- Trade-off:
  - The Flask app depends on both a reachable MLflow server and a local vectorizer file, which adds setup friction.
- Trade-off:
  - The dataset source is external, so reproducibility depends on that URL remaining available unless you vendor the dataset.
