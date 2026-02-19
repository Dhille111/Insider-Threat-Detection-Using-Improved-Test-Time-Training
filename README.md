# Insider Threat Detection — Deployment-ready Demo

This repository is a minimal, deployment-ready Python project demonstrating an insider-threat detection pipeline with an example Test-Time Training (TTT) adaptation endpoint.

Quick start (development):

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Train a demo model (uses synthetic data):

```bash
python -m src.train --epochs 5 --out artifacts/model.pt
```

3. Run the FastAPI server:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

4. Example request:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0]}'
```

Files of interest:

- `src/train.py`: training script that saves `artifacts/model.pt` and `artifacts/scaler.joblib`.
- `src/serve.py`: FastAPI server exposing `/predict` and `/ttt` endpoints.
- `Dockerfile`: container deployment.
