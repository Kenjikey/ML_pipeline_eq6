from mibici.monitoring.drift import (
    compute_psi,
    evaluate_drift_and_degradation,
    simulate_synthetic_drift,
)
from mibici.pipeline import ALL_FEATURE_COLS, TARGET_COLS, create_full_pipeline
from tests.test_modeling import _build_synthetic_features


def test_drift_detection_and_alerts():
    df_ref = _build_synthetic_features(200)
    df_ref["es_hora_pico"] = 1

    pipe = create_full_pipeline({"max_iter": 15, "max_depth": 4, "random_state": 42})
    pipe.fit(df_ref[ALL_FEATURE_COLS], df_ref[TARGET_COLS])

    # Sin drift: PSI bajo y sin alertas de drift
    same_report = evaluate_drift_and_degradation(df_ref, df_ref.copy(), pipe)
    assert same_report["status"] == "OK"
    assert len(same_report["alerts"]) == 0

    # Con drift sintetico: PSI > 0.20 y alertas disparadas
    df_drifted = simulate_synthetic_drift(df_ref)
    assert (
        compute_psi(df_ref["entradas_lag_1h"].values, df_drifted["entradas_lag_1h"].values) > 0.20
    )
    drift_report = evaluate_drift_and_degradation(df_ref, df_drifted, pipe)
    assert drift_report["status"] == "ALERT"
    assert len(drift_report["alerts"]) > 0
