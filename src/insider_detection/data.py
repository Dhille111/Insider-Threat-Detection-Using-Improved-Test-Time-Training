"""Data utilities: synthetic dataset for demo and simple loader."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_synthetic_data(n_samples=2000, n_features=20, n_classes=2, test_size=0.2, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.6),
                               n_redundant=int(n_features*0.1), n_classes=n_classes, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train.astype(np.float32), y_train.astype(np.int64), X_val.astype(np.float32), y_val.astype(np.int64), scaler
