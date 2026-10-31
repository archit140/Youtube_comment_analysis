"""Shared dataset validation and split helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COLUMN = "clean_comment"
LABEL_COLUMN = "category"
LABELS = (-1, 0, 1)
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the minimum labeled-comment dataset contract."""
    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = df[[TEXT_COLUMN, LABEL_COLUMN] + [
        c for c in df.columns if c not in required
    ]].copy()
    cleaned = cleaned.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    cleaned[TEXT_COLUMN] = cleaned[TEXT_COLUMN].astype(str).str.strip()
    cleaned[LABEL_COLUMN] = pd.to_numeric(cleaned[LABEL_COLUMN], errors="raise").astype(int)
    cleaned = cleaned[cleaned[TEXT_COLUMN].ne("")]
    invalid = sorted(set(cleaned[LABEL_COLUMN]) - set(LABELS))
    if invalid:
        raise ValueError(f"Unsupported sentiment labels: {invalid}; expected {list(LABELS)}")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if cleaned[LABEL_COLUMN].nunique() < len(LABELS):
        raise ValueError("Dataset must contain all three sentiment classes.")
    return cleaned


def stratified_split(
    df: pd.DataFrame, *, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic train/test partitions while preserving labels."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def label_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {str(label): int((df[LABEL_COLUMN] == label).sum()) for label in LABELS}
