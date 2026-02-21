"""FastAPI service exposing predict and TTT endpoints."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Insider Threat Detection Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .status { padding: 15px; border-radius: 4px; margin: 20px 0; font-weight: bold; }
        .status.ready { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .section { margin: 30px 0; }
        input, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { padding: 10px 20px; margin: 10px 0; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: #f9f9f9; padding: 15px; border-left: 4px solid #007bff; margin: 15px 0; border-radius: 4px; }
        .result pre { overflow-x: auto; }
        .feature-gen { font-size: 0.9em; color: #666; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Insider Threat Detection System</h1>
        <p>Test-Time Training (TTT) powered by scikit-learn MLPClassifier</p>
        
        <div id="statusDiv" class="status error">
            <span id="statusText">Checking model status...</span>
        </div>

        <div class="section">
            <h2>Quick Test</h2>
            <p>Generate random features and test the model:</p>
            <button onclick="generateAndPredict()">Generate Random Features & Predict</button>
            <div id="quickResult"></div>
        </div>

        <div class="section">
            <h2>Manual Prediction</h2>
            <p>Enter 20 comma-separated feature values (or paste JSON):</p>
            <textarea id="featuresInput" rows="4" placeholder="0.5, 0.3, 0.1, ... (need 20 values)"></textarea>
            <button onclick="predictManual()">Predict</button>
            <button onclick="applyTTT()">Predict + Apply TTT</button>
            <div id="manualResult"></div>
        </div>

        <div class="section">
            <h2>API Endpoints</h2>
            <ul>
                <li><code>GET /health</code> - Model status</li>
                <li><code>POST /predict</code> - Single prediction</li>
                <li><code>POST /ttt?steps=5</code> - Predict with test-time training</li>
                <li><code>GET /docs</code> - Full API documentation</li>
            </ul>
        </div>
    </div>

    <script>
        async function checkStatus() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                const statusDiv = document.getElementById('statusDiv');
                const statusText = document.getElementById('statusText');
                if (data.ready) {
                    statusDiv.className = 'status ready';
                    statusText.textContent = '✓ Model Ready';
                } else {
                    statusDiv.className = 'status error';
                    statusText.textContent = '✗ Model Not Ready (train first)';
                }
            } catch (e) {
                document.getElementById('statusDiv').className = 'status error';
                document.getElementById('statusText').textContent = '✗ Cannot reach API';
            }
        }

        async function generateAndPredict() {
            const features = Array.from({length: 20}, () => Math.random()).map(x => x.toFixed(4));
            document.getElementById('featuresInput').value = features.join(', ');
            await predictManual();
        }

        async function predictManual() {
            const input = document.getElementById('featuresInput').value.trim();
            let features;
            try {
                if (input.startsWith('[')) {
                    features = JSON.parse(input);
                } else {
                    features = input.split(',').map(x => parseFloat(x.trim()));
                }
                if (features.length !== 20 || features.some(isNaN)) throw new Error('Need exactly 20 numeric values');
            } catch (e) {
                document.getElementById('manualResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><pre>Error: ' + e.message + '</pre></div>';
                return;
            }
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({features})
                });
                const result = await res.json();
                document.getElementById('manualResult').innerHTML = '<div class="result"><strong>Prediction:</strong><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
            } catch (e) {
                document.getElementById('manualResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><pre>Error: ' + e.message + '</pre></div>';
            }
        }

        async function applyTTT() {
            const input = document.getElementById('featuresInput').value.trim();
            let features;
            try {
                if (input.startsWith('[')) {
                    features = JSON.parse(input);
                } else {
                    features = input.split(',').map(x => parseFloat(x.trim()));
                }
                if (features.length !== 20 || features.some(isNaN)) throw new Error('Need exactly 20 numeric values');
            } catch (e) {
                document.getElementById('manualResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><pre>Error: ' + e.message + '</pre></div>';
                return;
            }
            try {
                const res = await fetch('/ttt?steps=5', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({features})
                });
                const result = await res.json();
                document.getElementById('manualResult').innerHTML = '<div class="result"><strong>TTT Prediction:</strong><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
            } catch (e) {
                document.getElementById('manualResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><pre>Error: ' + e.message + '</pre></div>';
            }
        }

        checkStatus();
        setInterval(checkStatus, 5000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_DASHBOARD


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
