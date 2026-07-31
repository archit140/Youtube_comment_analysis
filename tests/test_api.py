import unittest
from unittest.mock import patch

from flask_app.app import app


class DummyVectorizer:
    def transform(self, comments):
        return [[len(comment)] for comment in comments]


class DummyModel:
    def predict(self, transformed):
        return [1 for _ in transformed]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("flask_app.app.get_model_vectorizer", return_value=(DummyModel(), DummyVectorizer()))
    def test_predict_returns_predictions(self, _mock_loader):
        response = self.client.post("/predict", json={"comments": ["great video", "bad ending"]})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["sentiment"], "1")

    def test_predict_requires_comments(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)

    def test_chart_requires_counts(self):
        response = self.client.post("/generate_chart", json={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
