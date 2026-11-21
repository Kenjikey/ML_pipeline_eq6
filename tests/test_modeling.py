import joblib
import numpy as np
import pandas as pd
import pytest

from mibici.modeling.evaluate import (
    BaselineHeuristicModel,
    calculate_metrics,
    evaluate_net_balance,
)
from mibici.modeling.predict import classify_rebalancing_action, load_pipeline
from mibici.modeling.train import select_by_validation, split_out_of_time
from mibici.pipeline import ALL_FEATURE_COLS, TARGET_COLS, create_full_pipeline


def _build_synthetic_features(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(0.0, 10.0, size=n) for col in ALL_FEATURE_COLS}
    data["estacion_id"] = rng.choice([51, 52, 100], size=n)
    # Garantizar presencia de Ene-Sep (train), Oct (val) y Nov-Dic (test)
    months = np.resize([1, 5, 9, 10, 11, 12], n)
    data["mes"] = months
    data["dia_mes"] = rng.integers(1, 29, size=n)
    data["dia_semana"] = rng.integers(0, 7, size=n)
    data["hora"] = rng.integers(0, 24, size=n)
    data["es_fin_de_semana"] = (data["dia_semana"] >= 5).astype(int)
    data["es_hora_pico"] = np.isin(data["hora"], [7, 8, 9, 18, 19, 20]).astype(int)
    data["entradas"] = rng.integers(0, 15, size=n)
    data["salidas"] = rng.integers(0, 15, size=n)
    data["balance"] = data["entradas"] - data["salidas"]
    return pd.DataFrame(data)


def test_pipeline_fit_serialization_unknown_station_and_reproducibility(tmp_path):
    df = _build_synthetic_features(180)
    X, y = df[ALL_FEATURE_COLS], df[TARGET_COLS]

    params = {"max_iter": 15, "max_depth": 4, "random_state": 42}
    pipe_a = create_full_pipeline(params).fit(X, y)
    pipe_b = create_full_pipeline(params).fit(X, y)

    # Reproducibilidad exacta con semilla fija
    np.testing.assert_allclose(pipe_a.predict(X), pipe_b.predict(X))

    artifact = tmp_path / "mibici_balance_pipeline.joblib"
    joblib.dump(pipe_a, artifact)
    loaded = load_pipeline(tmp_path)

    # Lote de 1 fila con estacion no vista en entrenamiento (9999)
    single_row = X.iloc[[0]].copy()
    single_row["estacion_id"] = 9999
    pred_one = np.clip(loaded.predict(single_row), 0, None)
    assert pred_one.shape == (1, 2)
    assert np.isfinite(pred_one).all()

    # Lote de N filas
    preds = np.clip(loaded.predict(X), 0, None)
    assert preds.shape == (len(df), 2)
    assert not np.isnan(preds).any()

    m_ent = calculate_metrics(y["entradas"].values, preds[:, 0], prefix="ent")
    m_bal = evaluate_net_balance(
        y["entradas"].values, preds[:, 0], y["salidas"].values, preds[:, 1]
    )
    assert m_ent["ent_mae"] >= 0.0
    assert "balance_rmse" in m_bal


def test_temporal_split_and_validation_selection():
    df = _build_synthetic_features(180)
    train_df, val_df, test_df = split_out_of_time(df)

    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    assert (train_df["mes"] <= 9).all()
    assert (val_df["mes"] == 10).all()
    assert (test_df["mes"] >= 11).all()
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    candidates = [
        {"model_family": "HGB", "max_iter": 10, "max_depth": 3, "random_state": 42},
        {"model_family": "HGB", "max_iter": 20, "max_depth": 5, "random_state": 42},
    ]
    best_pipe, best_params, best_val = select_by_validation(
        train_df[ALL_FEATURE_COLS],
        train_df[TARGET_COLS],
        val_df[ALL_FEATURE_COLS],
        val_df[TARGET_COLS],
        candidates,
    )
    assert best_pipe is not None
    assert "max_iter" in best_params
    assert "val_balance_mae" in best_val


def test_evaluate_edge_cases_baseline_and_rebalancing_actions(tmp_path):
    # Recorte de predicciones negativas en calculate_metrics
    m = calculate_metrics(np.array([0.0, 4.0]), np.array([-3.0, 4.0]), prefix="t")
    assert m["t_mae"] == 0.0

    # BaselineHeuristicModel con categoria desconocida -> usa global_median
    df = _build_synthetic_features(120)
    heur = BaselineHeuristicModel().fit(df, target_col="entradas")
    unseen = df.iloc[[0]].copy()
    unseen["estacion_id"] = 8888
    pred_unseen = heur.predict(unseen)
    assert pred_unseen[0] == pytest.approx(heur.global_median)

    # Reglas de accion operativa (+/- 2.0)
    balances = np.array([2.5, 2.0, 0.0, -1.9, -2.0, -4.1])
    actions = classify_rebalancing_action(balances)
    assert actions[0] == "ALERTA SUPERAVIT (Retirar Bicis)"
    assert actions[1] == "ALERTA SUPERAVIT (Retirar Bicis)"
    assert actions[2] == "ESTABLE"
    assert actions[3] == "ESTABLE"
    assert actions[4] == "ALERTA DEFICIT (Abastecer Bicis)"
    assert actions[5] == "ALERTA DEFICIT (Abastecer Bicis)"

    # Error explicito si no existe el joblib
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "inexistente")
