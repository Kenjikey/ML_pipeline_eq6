import numpy as np
import pandas as pd

from mibici.monitoring.drift import (
    MONITORED_FEATURES,
    compute_psi,
    evaluate_drift_and_degradation,
    simulate_synthetic_drift,
)
from mibici.pipeline import ALL_FEATURE_COLS, TARGET_COLS, create_full_pipeline


def _build_reference_dataset(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(1.0, 8.0, size=n) for col in ALL_FEATURE_COLS}
    data["estacion_id"] = rng.choice([51, 52, 100], size=n)
    data["mes"] = 10
    data["dia_mes"] = rng.integers(1, 32, size=n)
    data["dia_semana"] = rng.integers(0, 7, size=n)
    data["hora"] = rng.integers(0, 24, size=n)
    data["es_fin_de_semana"] = (data["dia_semana"] >= 5).astype(int)
    data["es_hora_pico"] = np.isin(data["hora"], [7, 8, 9, 18, 19, 20]).astype(int)

    df = pd.DataFrame(data)
    df["entradas"] = np.round(df["entradas_lag_1h"] * 0.8 + rng.uniform(0.0, 2.0, size=n))
    df["salidas"] = np.round(df["salidas_lag_1h"] * 0.8 + rng.uniform(0.0, 2.0, size=n))
    df["balance"] = df["entradas"] - df["salidas"]
    return df


def test_compute_psi_stable_shifted_and_constant():
    rng = np.random.default_rng(42)
    base = rng.normal(loc=10.0, scale=2.0, size=2000)
    same_dist = rng.normal(loc=10.0, scale=2.0, size=2000)
    shifted_dist = rng.normal(loc=16.0, scale=2.5, size=2000)

    psi_stable = compute_psi(base, same_dist)
    psi_shifted = compute_psi(base, shifted_dist)

    assert psi_stable < 0.05
    assert psi_shifted >= 0.20

    # Columna constante (< 3 cuantiles unicos) retorna 0.0 sin fallar
    constant = np.ones(100)
    assert compute_psi(constant, constant) == 0.0


def test_evaluate_drift_ok_vs_synthetic_drift_alert():
    df_ref = _build_reference_dataset(300)
    pipe = create_full_pipeline({"max_iter": 20, "max_depth": 4, "random_state": 42})
    pipe.fit(df_ref[ALL_FEATURE_COLS], df_ref[TARGET_COLS])

    # Escenario 1: Sin drift (misma muestra -> status OK)
    report_ok = evaluate_drift_and_degradation(df_ref, df_ref.copy(), pipe)
    assert report_ok["status"] == "OK"
    assert report_ok["alerts"] == []
    assert set(report_ok["feature_drift"].keys()) == set(MONITORED_FEATURES)

    # Escenario 2: Drift sintetico (+65% hora pico + concept drift -> status ALERT)
    df_drifted = simulate_synthetic_drift(df_ref, seed=42)
    report_alert = evaluate_drift_and_degradation(df_ref, df_drifted, pipe)

    assert report_alert["status"] == "ALERT"
    assert len(report_alert["alerts"]) >= 1
    assert any("DATA_DRIFT" in msg for msg in report_alert["alerts"])
    assert report_alert["sample_sizes"] == {"reference": 300, "current": 300}
    assert (
        report_alert["current_metrics"]["balance_mae"]
        >= report_alert["reference_metrics"]["balance_mae"]
    )
