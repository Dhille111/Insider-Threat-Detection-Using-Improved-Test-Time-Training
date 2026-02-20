"""FastAPI service exposing predict and TTT endpoints."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from .insider_detection.model import load_model, ThreatModel
from .insider_detection.ttt import adapt_entropy_minimization

app = FastAPI(title="Insider Threat Detection (demo)")


class PredictRequest(BaseModel):
    features: list


MODEL_PATH = "artifacts/model.joblib"
SCALER_PATH = "artifacts/scaler.joblib"


def load_artifacts():
    scaler = None
    model = None
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        pass
    try:
        model = load_model(MODEL_PATH, n_features=20)
    except Exception:
        model = None
    return model, scaler


model, scaler = load_artifacts()


@app.get("/health")
def health():
    return {"ready": model is not None and scaler is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available. Train and save first.")
    x = np.array(req.features, dtype=np.float32).reshape(1, -1)
    x = scaler.transform(x)
    probs = model.predict_proba(x)[0].tolist()
    pred = int(np.argmax(probs))
    return {"pred": pred, "probs": probs}


@app.post("/ttt")
def ttt(req: PredictRequest, steps: int = 5):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available. Train and save first.")
    x = np.array(req.features, dtype=np.float32).reshape(1, -1)
    x = scaler.transform(x)
    adapt_entropy_minimization(model, x, steps=steps)
    probs = model.predict_proba(x)[0].tolist()
    pred = int(np.argmax(probs))
    return {"pred": pred, "probs": probs}
