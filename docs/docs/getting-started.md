# Getting Started

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optionally copy the environment template:

```bash
copy .env.example .env
```

## Run The Pipeline

```bash
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

If you use DVC:

```bash
dvc repro
```

## Run The API

```bash
python flask_app/app.py
```

The default local endpoint is `http://127.0.0.1:5000`.
