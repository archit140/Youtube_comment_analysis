import unittest
import importlib.util
from pathlib import Path

from src.data.contracts import stratified_split, validate_dataset
from src.models import available_models, available_traditional_models, build_model
from project_utils import PROJECT_ROOT


class ModelTests(unittest.TestCase):
    def test_model_factory_lists_expected_models(self):
        self.assertEqual(
            available_traditional_models(),
            ("logistic_regression", "naive_bayes", "linear_svm", "lightgbm"),
        )
        self.assertIn("mlp", available_models())

    def test_traditional_models_fit_tiny_dataset(self):
        texts = ["good video", "great explanation", "bad video", "terrible audio", "okay", "fine"]
        labels = [1, 1, -1, -1, 0, 0]
        for name in available_traditional_models():
            with self.subTest(model=name):
                model = build_model(name, max_features=20, ngram_range=(1, 1))
                model.fit(texts, labels)
                self.assertEqual(len(model.predict(["good explanation"])), 1)

    def test_contract_validates_and_splits_stratified(self):
        import pandas as pd

        frame = pd.DataFrame(
            {"clean_comment": ["a", "b", "c", "d", "e", "f"], "category": [-1, -1, 0, 0, 1, 1]}
        )
        cleaned = validate_dataset(frame)
        train, test = stratified_split(cleaned, test_size=0.5, random_state=42)
        self.assertEqual(set(train["category"]), {-1, 0, 1})
        self.assertEqual(set(test["category"]), {-1, 0, 1})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional dependency")
    def test_mlp_tiny_training_and_probabilities(self):
        from src.models.mlp import TextMlpClassifier

        model = TextMlpClassifier(max_features=20, hidden_size=8, epochs=2, batch_size=2)
        model.fit(["good", "bad", "okay", "great", "awful", "fine"], [1, -1, 0, 1, -1, 0])
        probabilities = model.predict_proba(["good"])
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional dependency")
    def test_mlp_save_load_round_trip(self):
        from src.models.mlp import TextMlpClassifier

        model = TextMlpClassifier(max_features=20, hidden_size=8, epochs=2, batch_size=2)
        texts = ["good", "bad", "okay", "great", "awful", "fine"]
        labels = [1, -1, 0, 1, -1, 0]
        model.fit(texts, labels)
        path = PROJECT_ROOT / "models" / "_mlp_roundtrip_test.pt"
        try:
            model.save(path)
            restored = TextMlpClassifier.load(path)
            self.assertEqual(model.predict(["good"])[0], restored.predict(["good"])[0])
        finally:
            if path.exists():
                path.unlink()

    def test_local_storage_configuration(self):
        from pathlib import Path

        dvc_config = Path(".dvc/config").read_text(encoding="utf-8").lower()
        env_template = Path(".env.example").read_text(encoding="utf-8").lower()
        self.assertIn("url = storage", dvc_config)
        self.assertNotIn("s3", dvc_config)
        self.assertIn("sqlite:///./mlflow.db", env_template)
        self.assertNotIn("ec2", env_template)


if __name__ == "__main__":
    unittest.main()
