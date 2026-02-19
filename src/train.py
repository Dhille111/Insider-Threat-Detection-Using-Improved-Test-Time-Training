"""Train a simple model on synthetic data and save artifacts.

Usage:
    python -m src.train --epochs 5 --out artifacts/model.pt
"""
import argparse
from insider_detection.data import load_synthetic_data
from insider_detection.model import ThreatModel, save_model_obj
import joblib
import numpy as np


def train(args):
    X_train, y_train, X_val, y_val, scaler = load_synthetic_data(n_features=args.n_features)
    model = ThreatModel(n_features=args.n_features, n_classes=args.n_classes, hidden=args.hidden)
    model.fit(X_train, y_train, epochs=args.epochs)
    # save model and scaler
    save_model_obj(model, args.out)
    joblib.dump(scaler, args.scaler_out)
    print("Saved model and scaler.")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-features", dest="n_features", type=int, default=20)
    p.add_argument("--n-classes", dest="n_classes", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out", type=str, default="artifacts/model.joblib")
    p.add_argument("--scaler-out", type=str, default="artifacts/scaler.joblib")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
