# Project Timeline

## Original Project Work

The original project already included the core end-to-end machine learning workflow:

- DVC pipeline stages for ingestion, preprocessing, training, evaluation, and registration
- TF-IDF feature engineering
- LightGBM sentiment classification
- MLflow experiment tracking and model registration
- Flask endpoints for prediction and visualization
- Experiment notebooks for comparing approaches

## Completed During This Review

This review pass focused on interview readiness and GitHub presentation rather than turning the project into a heavyweight production system.

- Replaced the generic template README with project-specific documentation
- Added `ARCHITECTURE.md` explaining components, responsibilities, and data flow
- Added `INTERVIEW_NOTES.md` to convert the project into a structured revision guide
- Added `PROJECT_TIMELINE.md` to separate original work from review cleanup
- Added `.env.example` so the Flask app configuration is easier to understand
- Promoted the more usable Flask app version and removed the duplicate `app1.py`
- Centralized preprocessing, paths, and logging in `project_utils.py`
- Removed scaffold-only dead code that no longer matched the real implementation
- Added unit tests for preprocessing and API validation
- Added `Procfile`, `wsgi.py`, and `Dockerfile` so the project is deploy-ready

## Remaining Optional Improvements

- Add automated tests for preprocessing and API payload validation
- Refactor duplicated preprocessing into one shared module
- Trim dependencies to the minimal required set
- Add a short demo script or Postman collection for API usage
- Separate local experiment artifacts from repository-tracked source even more clearly
