from fastapi.testclient import TestClient
import joblib
import pytest

from mibici.pipeline import ALL_FEATURE_COLS, TARGET_COLS, create_full_pipeline
import mibici.serving.app as api_module
from tests.test_modeling import _build_synthetic_features


@pytest.fixture
def client(tmp_path, monkeypatch):
    df = _build_synthetic_features(80)
    pipe = create_full_pipeline({"max_iter": 10, "max_depth": 3, "random_state": 42})
    pipe.fit(df[ALL_FEATURE_COLS], df[TARGET_COLS])

    model_file = tmp_path / "mibici_balance_pipeline.joblib"
    joblib.dump(pipe, model_file)
    monkeypatch.setattr(api_module, "MODEL_PATH", model_file)

    with TestClient(api_module.app) as c:
        yield c


def _valid_payload() -> dict:
    return {
        "estacion_id": 51,
        "mes": 11,
        "dia_mes": 15,
        "dia_semana": 2,
        "hora": 8,
        "es_fin_de_semana": 0,
        "es_hora_pico": 1,
        "hora_sin": 0.866,
        "hora_cos": -0.5,
        "dia_semana_sin": 0.9749,
        "dia_semana_cos": -0.2225,
        "entradas_lag_1h": 6.0,
        "entradas_lag_2h": 4.0,
        "entradas_lag_24h": 7.0,
        "entradas_lag_168h": 5.0,
        "salidas_lag_1h": 3.0,
        "salidas_lag_2h": 2.0,
        "salidas_lag_24h": 4.0,
        "salidas_lag_168h": 3.0,
        "balance_lag_1h": 3.0,
        "balance_lag_2h": 2.0,
        "balance_lag_24h": 3.0,
        "balance_lag_168h": 2.0,
        "entradas_roll_mean_3h": 4.33,
        "entradas_roll_mean_6h": 3.5,
        "salidas_roll_mean_3h": 2.33,
        "salidas_roll_mean_6h": 2.0,
        "balance_roll_mean_3h": 2.0,
        "balance_roll_mean_6h": 1.5,
    }


def test_health_and_schema(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["expected_features"] == 29


def test_predict_contract_valid_and_invalid(client):
    ok = client.post("/predict", json=_valid_payload())
    assert ok.status_code == 200
    body = ok.json()
    assert set(body.keys()) == {
        "estacion_id",
        "pred_entradas",
        "pred_salidas",
        "pred_balance",
        "accion_operativa",
    }

    # Campo extra prohibido o fuera de rango debe retornar 422
    bad_hour = {**_valid_payload(), "hora": 25}
    assert client.post("/predict", json=bad_hour).status_code == 422

    extra_field = {**_valid_payload(), "columna_inventada": 123}
    assert client.post("/predict", json=extra_field).status_code == 422
