import unittest
import mimetypes
from io import BytesIO
from unittest.mock import patch

from flask_app.app import app


# Avoid reading the protected Windows MIME registry in restricted test hosts.
mimetypes.inited = True
mimetypes.types_map = {".csv": "text/csv"}
mimetypes._db = mimetypes.MimeTypes(filenames=[])
mimetypes._db.add_type("text/csv", ".csv")


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
        self.assertEqual(data[0]["model"], "DummyModel")

    @patch("flask_app.app.get_model_vectorizer", return_value=(DummyModel(), DummyVectorizer()))
    def test_predict_accepts_single_comment(self, _mock_loader):
        response = self.client.post("/predict", json={"comment": "great video"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["comment"], "great video")
        self.assertEqual(data["sentiment"], "1")

    def test_predict_rejects_invalid_json_shape(self):
        response = self.client.post("/predict", json={"comments": "not a list"})
        self.assertEqual(response.status_code, 400)

    def test_predict_requires_comments(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)

    def test_chart_requires_counts(self):
        response = self.client.post("/generate_chart", json={})
        self.assertEqual(response.status_code, 400)

    def test_csv_requires_file(self):
        response = self.client.post("/predict_csv")
        self.assertEqual(response.status_code, 400)

    @patch("flask_app.app.get_model_vectorizer", return_value=(DummyModel(), DummyVectorizer()))
    def test_csv_returns_predictions(self, _mock_loader):
        response = self.client.post(
            "/predict_csv",
            data={"file": (BytesIO(b"comment\ngreat video\nbad ending\n"), "comments.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"sentiment", response.data)

    def test_csv_requires_supported_column(self):
        response = self.client.post(
            "/predict_csv",
            data={"file": (BytesIO(b"wrong\nvalue\n"), "comments.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
