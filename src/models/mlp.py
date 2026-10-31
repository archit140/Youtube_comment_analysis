"""Small CPU-friendly PyTorch MLP over TF-IDF features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextMlpClassifier:
    def __init__(
        self,
        *,
        max_features: int = 5000,
        hidden_size: int = 128,
        dropout: float = 0.2,
        epochs: int = 15,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.classes_ = np.array([-1, 0, 1])
        self.network = None

    def _require_torch(self):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for --model mlp. Install torch first.") from exc
        return torch, nn

    def fit(self, texts, labels, validation_data=None):
        torch, nn = self._require_torch()
        torch.manual_seed(self.random_state)
        features = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        x = torch.tensor(features)
        y = torch.tensor([int(label) + 1 for label in labels], dtype=torch.long)
        self.network = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 3),
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        validation_tensors = None
        if validation_data is not None:
            valid_texts, valid_labels = validation_data
            valid_features = self.vectorizer.transform(valid_texts).toarray().astype("float32")
            validation_tensors = (
                torch.tensor(valid_features),
                torch.tensor([int(label) + 1 for label in valid_labels], dtype=torch.long),
            )

        best_state = None
        best_loss = float("inf")
        patience = 3
        stale_epochs = 0
        for _ in range(self.epochs):
            self.network.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.network(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            if validation_tensors is not None:
                self.network.eval()
                with torch.no_grad():
                    valid_x, valid_y = validation_tensors
                    current_loss = float(criterion(self.network(valid_x), valid_y).item())
            else:
                current_loss = float(loss.detach().item())
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def _probabilities(self, texts):
        torch, _ = self._require_torch()
        if self.network is None:
            raise RuntimeError("The MLP has not been trained.")
        self.network.eval()
        features = self.vectorizer.transform(texts).toarray().astype("float32")
        with torch.no_grad():
            return torch.softmax(self.network(torch.tensor(features)), dim=1).numpy()

    def predict_proba(self, texts):
        return self._probabilities(texts)

    def predict(self, texts):
        indices = np.argmax(self._probabilities(texts), axis=1)
        return self.classes_[indices]

    def save(self, path: Path) -> None:
        torch, _ = self._require_torch()
        if self.network is None:
            raise RuntimeError("The MLP has not been trained.")
        torch.save({"model": self, "state_dict": self.network.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "TextMlpClassifier":
        torch, _ = cls()._require_torch()
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            checkpoint = torch.load(path, map_location="cpu")
        model = checkpoint["model"]
        model.network.load_state_dict(checkpoint["state_dict"])
        return model
