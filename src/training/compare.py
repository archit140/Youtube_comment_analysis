"""Train selected models and write a measured comparison report."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from project_utils import PROJECT_ROOT
from src.models import available_models
from src.training.train import train


def compare(models: tuple[str, ...], random_state: int = 42) -> pd.DataFrame:
    results = [train(name, random_state=random_state) for name in models]
    rows = []
    for result in results:
        test = result["test"]
        rows.append({
            "model_name": result["model_name"],
            "accuracy": test["accuracy"],
            "macro_precision": test["macro_precision"],
            "macro_recall": test["macro_recall"],
            "macro_f1": test["macro_f1"],
            "weighted_f1": test["weighted_f1"],
            "training_seconds": result["training_seconds"],
            "inference_seconds": test["inference_seconds"],
            "model_size_bytes": result["model_size_bytes"],
            "supports_probability": result["supports_probability"],
            "negative_f1": test["per_class"]["-1"]["f1-score"],
            "neutral_f1": test["per_class"]["0"]["f1-score"],
            "positive_f1": test["per_class"]["1"]["f1-score"],
            "negative_recall": test["per_class"]["-1"]["recall"],
            "neutral_recall": test["per_class"]["0"]["recall"],
            "positive_recall": test["per_class"]["1"]["recall"],
        })
    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            result["test"]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"],
        )
        plt.title(f"{result['model_name']} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{result['model_name']}_confusion_matrix.png")
        plt.close()
    frame.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "model_comparison.json").write_text(
        json.dumps(
            {"models": rows, "selection_metric": "macro_f1", "test_set_only": True},
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=available_models(), default=list(available_models()))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(compare(tuple(args.models), random_state=args.random_state).to_string(index=False))


if __name__ == "__main__":
    main()
