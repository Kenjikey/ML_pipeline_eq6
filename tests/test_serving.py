import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest

from mibici.pipeline import ALL_FEATURE_COLS, create_full_pipeline, get_feature_groups
import mibici.serving.app as app_module


def _sample_record(estacion_id: int = 51) -> dict:
    """Construye un registro valido con estacion_id y las 29 features."""
    record = {col: 1.0 for col in ALL_FEATURE_COLS}
    record.update(
        {
            "estacion_id": estacion_id,
            "mes": 11,
            "dia_mes": 15,
            "dia_semana": 2,
            "hora": 8,
            "es_fin_de_semana": 0,
            "es_hora_pico": 1,
        }
    )
    return record


def _build_dummy_pipeline():
    rng = np.random.default_rng(42)
    n = 40
    X = pd.DataFrame({col: rng.uniform(0.0, 5.0, size=n) for col in ALL_FEATURE_COLS})
    X["estacion_id"] = rng.choice([51, 52], size=n)
    for col in ["mes", "dia_mes", "dia_semana", "hora", "es_fin_de_semana", "es_hora_pico"]:
        X[col] = rng.integers(0, 2, size=n)
    y = pd.DataFrame(
        {
            "entradas": rng.integers(0, 10, size=n),
            "salidas": rng.integers(0, 10, size=n),
        }
    )
    return create_full_pipeline({"max_iter": 10, "max_depth": 3, "random_state": 42}).fit(X, y)


@pytest.fixture()
def client_with_model(tmp_path, monkeypatch):
    dummy_pipe = _build_dummy_pipeline()
    model_path = tmp_path / "mibici_balance_pipeline.joblib"
    schema_path = tmp_path / "model_schema.json"

    joblib.dump(dummy_pipe, model_path)
    schema_payload = {
        "model_name": "mibici_balance_predictor",
        "model_version": "1",
        "dataset_id": "features_2023_v2",
        "feature_groups": get_feature_groups(),
    }
    schema_path.write_text(json.dumps(schema_payload), encoding="utf-8")

    monkeypatch.setattr(app_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(app_module, "SCHEMA_PATH", schema_path)

    with TestClient(app_module.app) as client:
        yield client


def test_health_and_model_info_endpoints(client_with_model: TestClient):
    res_health = client_with_model.get("/health")
    assert res_health.status_code == 200
    body_health = res_health.json()
    assert body_health["status"] == "healthy"
    assert body_health["model_loaded"] is True

    res_info = client_with_model.get("/model-info")
    assert res_info.status_code == 200
    body_info = res_info.json()
    assert body_info["model_name"] == "mibici_balance_predictor"
    assert len(body_info["features"]) == 29


def test_predict_single_and_batch_return_200(client_with_model: TestClient):
    # Prediccion individual en POST /predict
    res_single = client_with_model.post("/predict", json=_sample_record(51))
    assert res_single.status_code == 200
    first = res_single.json()
    assert first["estacion_id"] == 51
    assert first["pred_entradas"] >= 0.0
    assert first["pred_salidas"] >= 0.0
    assert first["pred_balance"] == pytest.approx(
        first["pred_entradas"] - first["pred_salidas"], abs=1e-2
    )
    assert "accion_operativa" in first

    # Prediccion por lote en POST /predict/batch
    res_batch = client_with_model.post(
        "/predict/batch", json={"records": [_sample_record(51), _sample_record(52)]}
    )
    assert res_batch.status_code == 200
    batch_body = res_batch.json()
    assert batch_body["count"] == 2
    assert len(batch_body["predictions"]) == 2


def test_predict_rejects_invalid_contract_with_422(client_with_model: TestClient):
    # 1. Feature obligatoria faltante
    missing_feat = _sample_record(51)
    missing_feat.pop("entradas_lag_1h")
    assert client_with_model.post("/predict", json=missing_feat).status_code == 422

    # 2. Columna target prohibida (prevencion de leakage, extra="forbid")
    leakage_record = _sample_record(51)
    leakage_record["entradas"] = 10.0
    assert client_with_model.post("/predict", json=leakage_record).status_code == 422

    # 3. Identificador de estacion no positivo
    invalid_station = _sample_record(0)
    assert client_with_model.post("/predict", json=invalid_station).status_code == 422


def test_predict_rejects_oversized_batch(client_with_model: TestClient):
    oversized = [_sample_record(51) for _ in range(1005)]
    response = client_with_model.post("/predict/batch", json={"records": oversized})
    assert response.status_code in (413, 422)


def test_predict_returns_503_when_model_unavailable(client_with_model: TestClient):
    app_module.state["pipeline"] = None
    response = client_with_model.post("/predict", json=_sample_record(51))
    assert response.status_code == 503
