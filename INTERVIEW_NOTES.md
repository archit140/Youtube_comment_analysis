# Interview Notes

## How To Revise This Project

Revise this repo in the same order that data flows through it:

1. Dataset ingestion and train/test split
2. NLP preprocessing
3. TF-IDF feature engineering
4. LightGBM training
5. Model evaluation metrics
6. DVC pipeline tracking
7. MLflow experiment tracking and model registry
8. Flask inference API

## High Priority Topics

### NLP Preprocessing

Where used:

- `src/data/data_preprocessing.py`
- `flask_app/app.py`
- `project_utils.py`

Why used:

- Raw comments are noisy, inconsistent, and not directly useful for vectorization.

What to revise:

- lowercasing
- token filtering
- stopword removal
- lemmatization
- why negation words should sometimes be preserved

Common interview questions:

- Why do preprocessing choices matter in sentiment analysis?
- Why keep words like `not` and `no`?
- What are the downsides of aggressive stopword removal?

Follow-up questions:

- Would stemming change performance?
- How would emojis, slang, or URLs affect this pipeline?

### TF-IDF

Where used:

- `src/model/model_building.py`
- `src/model/model_evaluation.py`
- `flask_app/app.py`

Why used:

- TF-IDF gives a strong, interpretable baseline for text classification without deep learning.

What to revise:

- term frequency
- inverse document frequency
- sparse matrices
- n-grams
- vocabulary consistency between training and inference

Common interview questions:

- Why TF-IDF instead of bag-of-words?
- Why use trigram features?
- Why must you save the vectorizer separately?

Follow-up questions:

- What happens if inference uses a different vectorizer?
- What is the effect of `max_features`?

### LightGBM

Where used:

- `src/model/model_building.py`

Why used:

- It is efficient, strong on tabular/sparse features, and works well with TF-IDF style inputs.

What to revise:

- boosting intuition
- multiclass classification
- hyperparameters: `learning_rate`, `max_depth`, `n_estimators`
- class imbalance handling

Common interview questions:

- Why LightGBM for text features?
- What does `class_weight="balanced"` do?
- How do boosting models differ from random forests?

Follow-up questions:

- Would logistic regression be a better baseline?
- Why not use transformers here?

### MLflow

Where used:

- `src/model/model_evaluation.py`
- `src/model/register_model.py`
- `flask_app/app.py`

Why used:

- It tracks runs, stores artifacts, and supports model versioning.

What to revise:

- experiments vs runs
- logging parameters, metrics, and artifacts
- model registry concepts
- stage transitions

Common interview questions:

- What does MLflow add beyond saving a `.pkl` file?
- Why store the model in a registry?
- What is the difference between tracking and registry?

Follow-up questions:

- How would you promote a model from staging to production?
- What metadata would you log for better reproducibility?

### Flask API

Where used:

- `flask_app/app.py`

Why used:

- It gives a lightweight serving layer for prediction and visualization endpoints.

What to revise:

- request handling
- JSON validation
- loading startup dependencies
- returning image responses

Common interview questions:

- Why expose a model behind an API?
- What inputs does `/predict` expect?
- How do you handle model loading failure?

Follow-up questions:

- How would you scale this API?
- What would you cache?

## Medium Priority Topics

### DVC

Where used:

- `dvc.yaml`
- `dvc.lock`

Why used:

- It formalizes the ML workflow as reproducible stages with dependencies and outputs.

What to revise:

- stages
- dependencies
- outputs
- parameter tracking

Common interview questions:

- Why use DVC in an ML project?
- How is DVC different from Git?

### Train/Test Split And Reproducibility

Where used:

- `src/data/data_ingestion.py`

Why used:

- Model evaluation is not meaningful without separating train and test data.

What to revise:

- data leakage
- reproducible splits
- why `random_state=42` matters

Common interview questions:

- Why split before modeling?
- What is leakage?

### Evaluation Metrics

Where used:

- `src/model/model_evaluation.py`

Why used:

- Accuracy alone is not enough for multiclass sentiment problems.

What to revise:

- precision
- recall
- F1-score
- confusion matrix

Common interview questions:

- When would precision matter more than recall?
- How do you read a confusion matrix?

## Low Priority Topics

### Cookiecutter Data Science Template

Where used:

- repository structure
- legacy `yt_comment/` package

Why it matters:

- It explains why some scaffold files exist even though the main implementation moved into `src/`.

### Makefile And Project Tooling

Where used:

- `Makefile`
- `pyproject.toml`

Why it matters:

- Useful for understanding developer tooling, but not central to the ML story.

## Key Talking Points To Remember

- The strongest part of this project is the end-to-end workflow, not just the model.
- You can explain both experimentation and model serving from one codebase.
- The model is trained on a labeled sentiment dataset and then applied to YouTube comments, so be ready to explain that transfer honestly.
- The biggest engineering strength is combining DVC, MLflow, and Flask in a compact real-world project.
- The biggest deployment dependency is MLflow availability plus access to the saved vectorizer.
