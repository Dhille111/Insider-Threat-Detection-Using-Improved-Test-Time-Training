"""Fallback scikit-learn MLP wrapper so the demo runs without PyTorch.

This provides a `ThreatModel` with `fit`, `predict_proba`, `predict`, `save`, and
`load` methods. It uses `MLPClassifier` with `partial_fit` so we can implement
lightweight test-time adaptation without heavy dependencies.
"""
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
from pathlib import Path


class ThreatModel:
    def __init__(self, n_features: int, n_classes: int = 2, hidden: int = 64):
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = MLPClassifier(hidden_layer_sizes=(hidden, hidden), max_iter=1, warm_start=True)
        self._is_fitted = False

    def fit(self, X, y, epochs: int = 10):
        X = np.asarray(X)
        y = np.asarray(y)
        for _ in range(epochs):
            self.model.partial_fit(X, y, classes=np.arange(self.n_classes))
        self._is_fitted = True

    def predict_proba(self, X):
        X = np.asarray(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))

    @classmethod
    def load(cls, path: str, n_features: int, n_classes: int = 2):
        model = cls(n_features=n_features, n_classes=n_classes)
        model.model = joblib.load(path)
        model._is_fitted = True
        return model


def save_model_obj(model_obj: ThreatModel, path: str):
    model_obj.save(path)


def load_model(path: str, n_features: int, n_classes: int = 2):
    return ThreatModel.load(path, n_features=n_features, n_classes=n_classes)
