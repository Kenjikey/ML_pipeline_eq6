"""Servicio FastAPI para inferencia horaria de demanda y rebalanceo MiBici."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd

from mibici.modeling.predict import classify_rebalancing_action
from mibici.pipeline import ALL_FEATURE_COLS
from mibici.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionItem,
    StationHourFeatures,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/mibici_balance_pipeline.joblib"))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "models/model_schema.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "2.0.0")

state = {"pipeline": None, "metadata": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Artefacto de modelo no encontrado en: {MODEL_PATH}")
    state["pipeline"] = joblib.load(MODEL_PATH)
    if SCHEMA_PATH.exists():
        state["metadata"] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    yield
    state["pipeline"] = None


app = FastAPI(
    title="MiBici Rebalancing Prediction API",
    description="Prediccion horaria de entradas, salidas y balance neto por estacion MiBici.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _run_inference(records: list[StationHourFeatures]) -> list[PredictionItem]:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado en memoria.")

    df = pd.DataFrame([r.model_dump() for r in records])[ALL_FEATURE_COLS]
    preds = np.clip(pipeline.predict(df), 0, None)
    pred_ent = preds[:, 0]
    pred_sal = preds[:, 1]
    pred_bal = pred_ent - pred_sal
    actions = classify_rebalancing_action(pred_bal)

    return [
        PredictionItem(
            estacion_id=int(records[i].estacion_id),
            pred_entradas=round(float(pred_ent[i]), 2),
            pred_salidas=round(float(pred_sal[i]), 2),
            pred_balance=round(float(pred_bal[i]), 2),
            accion_operativa=str(actions[i]),
        )
        for i in range(len(records))
    ]


@app.get("/health")
def health():
    loaded = state.get("pipeline") is not None
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "model_version": MODEL_VERSION,
        "dataset_id": state["metadata"].get("dataset_id", "features_2023_v2"),
        "expected_features": len(ALL_FEATURE_COLS),
    }


@app.get("/model-info")
@app.get("/schema")
def contract_schema():
    return {
        "model_name": state["metadata"].get("model_name", "mibici_balance_predictor"),
        "model_version": MODEL_VERSION,
        "dataset_id": state["metadata"].get("dataset_id", "features_2023_v2"),
        "features": ALL_FEATURE_COLS,
        "metadata": state["metadata"],
    }


@app.post("/predict", response_model=PredictionItem)
def predict_single(payload: StationHourFeatures):
    return _run_inference([payload])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    items = _run_inference(payload.records)
    return BatchPredictionResponse(
        model_version=MODEL_VERSION,
        count=len(items),
        predictions=items,
    )
