# Local-only security policy

This project is intentionally local-only.

- DVC uses `.dvc/storage` as its local remote.
- MLflow uses the repository-local `mlruns/` directory.
- The application binds to localhost by default.
- No AWS SDK, AWS CLI, cloud credentials, cloud endpoints, or cloud deployment workflow is required.
- Dataset download is limited to the explicit `DATASET_URL` source during ingestion; all later stages use local files.

The working copy removed historical notebooks that contained cloud setup commands and credential material. Removing files from the current checkout does not remove them from Git history. Before publishing the repository, review history with a secret-scanning tool and rewrite history if necessary; rotate any credential that may ever have been active.

Do not add credentials to `.env`, notebooks, source files, MLflow artifacts, DVC storage, or Git history.
