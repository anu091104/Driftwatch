"""
FastAPI service: serves the current champion model and exposes drift checks.

Run:  uvicorn api.main:app --reload       then open http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.data import FEATURES, drift_state, make_batch
from src.detectors import make_detector

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

app = FastAPI(title="DriftWatch API",
              description="Loan-default model serving with built-in data-drift checks.",
              version="1.0")

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


class Applicant(BaseModel):
    age: float = Field(35, ge=18, le=100)
    income: float = Field(52000, gt=0)
    loan_amount: float = Field(18000, gt=0)
    credit_score: float = Field(690, ge=300, le=850)
    debt_ratio: float = Field(0.32, ge=0, le=1)
    employment_years: float = Field(6, ge=0)


class Batch(BaseModel):
    applicants: List[Applicant]


def _load():
    if not (MODELS / "champion.joblib").exists():
        raise HTTPException(503, "No model yet. Run `python -m src.demo` first.")
    model = joblib.load(MODELS / "champion.joblib")
    ref = pd.read_csv(MODELS / "reference.csv")
    info = json.loads((MODELS / "model_info.json").read_text())
    return model, ref, info


def _drift_report(ref, X):
    report = {}
    for name in ["PSI", "KS", "JS", "MMD"]:
        det = make_detector(name)
        det.set_reference(ref[FEATURES])
        drift, score, detail = det.check(X[FEATURES])
        if detail.startswith("feature #"):
            detail = FEATURES[int(detail[9:])]
        report[name] = {"drift": bool(drift), "score": round(score, 4), "most_shifted": detail}
    votes = sum(r["drift"] for r in report.values())
    return {"drift_detected": votes >= 2, "detectors_voting_drift": votes, "detectors": report,
            "note": "Feature-based checks only (no labels). Concept drift needs labels: see DDM in the benchmark."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (MODELS / "champion.joblib").exists()}


@app.get("/model")
def model_info():
    return _load()[2]


@app.post("/predict")
def predict(batch: Batch):
    model, _, info = _load()
    X = pd.DataFrame([a.model_dump() for a in batch.applicants])[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    return {"model_version": info["version"],
            "predictions": [{"default_probability": round(float(p), 4), "will_default": bool(p >= 0.5)} for p in proba]}


@app.post("/drift/check")
def drift_check(batch: Batch):
    """Send a batch of recent applicants (50+ recommended) to compare against the model's training data."""
    if len(batch.applicants) < 20:
        raise HTTPException(422, "Send at least 20 applicants for a meaningful drift test.")
    _, ref, _ = _load()
    X = pd.DataFrame([a.model_dump() for a in batch.applicants])
    return _drift_report(ref, X)


@app.get("/drift/simulate")
def drift_simulate(scenario: Literal["no_drift", "sudden_covariate", "gradual_covariate", "concept"] = "sudden_covariate", n: int = Query(500, ge=50, le=5000)):
    """Generate a batch from a drift scenario (as it looks 10 batches after drift starts) and run the drift checks."""
    _, ref, info = _load()
    shock, concept = drift_state(scenario, 40)
    X, _ = make_batch(np.random.default_rng(), n, shock, concept)
    out = _drift_report(ref, X)
    out["scenario"], out["model_version"] = scenario, info["version"]
    return out
