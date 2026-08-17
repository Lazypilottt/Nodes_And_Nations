"""
Simple FastAPI app to serve edge weight predictions using the trained model.

Run with:
    source venv/bin/activate
    uvicorn api.predict:app --reload --port 8001

POST /predict with JSON body:
{
  "origin_iso3": "IND",
  "dest_iso3": "USA",
  "year": 2020
}

Or POST with precomputed feature vector:
{
  "features": { "origin_log_gdp": 8.2, "dest_log_gdp": 10.1, ... }
}

Returns JSON with predicted weight (and predicted migrant stock = exp(pred)-1)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
EXPORTS_DIR = os.path.join(ROOT, "data", "exports")
PROCESSED = os.path.join(ROOT, "data", "processed")

MODEL_PATH = os.path.join(MODELS_DIR, "edge_weight_predictor.joblib")
FACTORS_PATH = os.path.join(PROCESSED, "factors_panel.csv")
COUNTRY_META_PATH = os.path.join(EXPORTS_DIR, "country_metadata.csv")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run notebooks/06_edge_weight_predictor.py first to train and save the model.")

saved = joblib.load(MODEL_PATH)
MODEL = saved["model"]
FEATURES = saved["features"]

# Load lookup tables to build features from origin/dest/year
FACTORS = pd.read_csv(FACTORS_PATH)
COUNTRY_META = pd.read_csv(COUNTRY_META_PATH) if os.path.exists(COUNTRY_META_PATH) else None

app = FastAPI(title="Nodes & Nations Edge Predictor")

class PredictRequest(BaseModel):
    origin_iso3: str | None = None
    dest_iso3: str | None = None
    year: int | None = None
    features: dict | None = None


@app.post("/predict")
def predict(req: PredictRequest):
    # If precomputed features given, use them directly
    if req.features:
        missing = [f for f in FEATURES if f not in req.features]
        if missing:
            raise HTTPException(status_code=400, detail={"missing_features": missing})
        X = pd.DataFrame([ {f: req.features[f] for f in FEATURES} ])
    else:
        if not (req.origin_iso3 and req.dest_iso3 and req.year):
            raise HTTPException(status_code=400, detail="Provide either 'features' or origin_iso3, dest_iso3, year")
        o = req.origin_iso3
        d = req.dest_iso3
        y = int(req.year)
        # Lookup in FACTORS
        fo = FACTORS[(FACTORS["iso3"] == o) & (FACTORS["year"] == y)].squeeze()
        fd = FACTORS[(FACTORS["iso3"] == d) & (FACTORS["year"] == y)].squeeze()
        if fo.empty or fd.empty:
            raise HTTPException(status_code=404, detail="Origin or destination factors not found for that year")
        row = {}
        # Build matching feature set used during training
        for f in FEATURES:
            if f.startswith("origin_"):
                base = f.replace("origin_", "")
                if base in fo.index:
                    if base == "gdp_per_capita":
                        row[f] = np.log(fo[base]) if fo[base] > 0 else None
                    elif base == "population":
                        row[f] = np.log(fo[base]) if fo[base] > 0 else None
                    else:
                        row[f] = fo.get(base, None)
                else:
                    row[f] = None
            elif f.startswith("dest_"):
                base = f.replace("dest_", "")
                if base in fd.index:
                    if base == "gdp_per_capita":
                        row[f] = np.log(fd[base]) if fd[base] > 0 else None
                    elif base == "population":
                        row[f] = np.log(fd[base]) if fd[base] > 0 else None
                    else:
                        row[f] = fd.get(base, None)
                else:
                    row[f] = None
            else:
                # relational features
                if f == "gdp_log_diff":
                    o_gdp = np.log(fo["gdp_per_capita"]) if fo["gdp_per_capita"] > 0 else 0.0
                    d_gdp = np.log(fd["gdp_per_capita"]) if fd["gdp_per_capita"] > 0 else 0.0
                    row[f] = o_gdp - d_gdp
                elif f == "pop_log_diff":
                    o_p = np.log(fo["population"]) if fo["population"] > 0 else 0.0
                    d_p = np.log(fd["population"]) if fd["population"] > 0 else 0.0
                    row[f] = o_p - d_p
                elif f == "same_continent":
                    if COUNTRY_META is not None:
                        o_cont = COUNTRY_META.loc[COUNTRY_META["iso3"] == o, "continent"].squeeze()
                        d_cont = COUNTRY_META.loc[COUNTRY_META["iso3"] == d, "continent"].squeeze()
                        row[f] = 1.0 if o_cont == d_cont else 0.0
                    else:
                        row[f] = 0.0
                elif f == "same_income_group":
                    if COUNTRY_META is not None:
                        o_inc = COUNTRY_META.loc[COUNTRY_META["iso3"] == o, "income_group"].squeeze()
                        d_inc = COUNTRY_META.loc[COUNTRY_META["iso3"] == d, "income_group"].squeeze()
                        row[f] = 1.0 if o_inc == d_inc else 0.0
                    else:
                        row[f] = 0.0
                elif f == "year":
                    row[f] = y
                else:
                    row[f] = None
        X = pd.DataFrame([row])

    # Ensure columns order
    X = X[FEATURES]

    # If any NaNs remain, reject
    if X.isna().any(axis=1).any():
        return {"error": "Computed features have missing values", "row": X.to_dict(orient="records")[0]}

    pred_log = MODEL.predict(X)[0]
    pred = float(np.expm1(pred_log))
    return {"predicted_log_weight": float(pred_log), "predicted_weight": pred}
