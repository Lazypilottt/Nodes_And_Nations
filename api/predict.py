"""
FastAPI service that serves bilateral migrant-stock predictions from the trained
edge-weight model.

Run with:
    source venv/bin/activate
    python -m uvicorn api.predict:app --reload --port 8001

POST /predict with JSON body:
{
  "origin_iso3": "IND",
  "dest_iso3": "USA",
  "year": 2020
}

Or POST with a precomputed feature map:
{
  "features": { "origin_log_gdp": 8.2, "dest_log_gdp": 10.1, ... }
}

Returns the predicted log1p weight and the predicted migrant stock (expm1 of it).
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.features import build_dyad_features, network_lags
from data.loader import (
    MODELS_DIR, load_country_metadata, load_factors, load_geography_dyadic, load_migration_long,
)

MODEL_PATH = MODELS_DIR / "edge_weight_predictor.joblib"

app = FastAPI(title="Nodes & Nations Edge Predictor", version="1.0.0")


class PredictRequest(BaseModel):
    origin_iso3: Optional[str] = Field(None, min_length=3, max_length=3)
    dest_iso3: Optional[str] = Field(None, min_length=3, max_length=3)
    year: Optional[int] = Field(None, ge=1990, le=2025)
    features: Optional[Dict[str, Optional[float]]] = None


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run notebooks/06_edge_weight_predictor.py first.",
        )
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"]


@lru_cache(maxsize=1)
def _lookups():
    return load_factors(), load_country_metadata(), load_geography_dyadic(), network_lags(load_migration_long())


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(req: PredictRequest):
    model, features = _model()

    if req.features:
        missing = [f for f in features if f not in req.features]
        if missing:
            raise HTTPException(status_code=422, detail={"missing_features": missing})
        X = pd.DataFrame([{f: req.features[f] for f in features}], dtype=float)
    else:
        if not (req.origin_iso3 and req.dest_iso3 and req.year):
            raise HTTPException(status_code=422, detail="Provide either 'features' or origin_iso3, dest_iso3 and year")
        origin, dest = req.origin_iso3.upper(), req.dest_iso3.upper()
        if origin == dest:
            raise HTTPException(status_code=422, detail="origin_iso3 and dest_iso3 must differ")

        factors, meta, geography, lags = _lookups()
        known = set(factors.loc[factors["year"] == req.year, "iso3"])
        unknown = [iso for iso in (origin, dest) if iso not in known]
        if unknown:
            raise HTTPException(status_code=404, detail=f"No factor data for {', '.join(unknown)} in {req.year}")

        pair = pd.DataFrame([{"origin_iso3": origin, "dest_iso3": dest, "year": req.year}])
        X = build_dyad_features(pair, factors, meta, geography=geography, lags=lags)[features]

    # Gradient boosting handles NaN natively, but report which inputs were absent.
    missing_inputs = [c for c in features if pd.isna(X.iloc[0][c])]
    pred_log = float(model.predict(X)[0])
    return {
        "predicted_log_weight": pred_log,
        "predicted_weight": float(np.expm1(pred_log)),
        "missing_inputs": missing_inputs,
    }
