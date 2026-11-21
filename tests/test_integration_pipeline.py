import joblib
import numpy as np
import pandas as pd

from mibici.dataset import build_hourly_grid, clean_trips
from mibici.features import (
    add_calendar_and_cyclical_features,
    add_historical_lags_and_rolling,
    validate_hourly_grid,
)
from mibici.modeling.evaluate import evaluate_net_balance
from mibici.modeling.predict import classify_rebalancing_action
from mibici.pipeline import ALL_FEATURE_COLS, TARGET_COLS, create_full_pipeline


def _build_raw_trips_fixture() -> pd.DataFrame:
    """Genera viajes crudos validos (enero 2023) mas registros sucios a descartar."""
    rng = np.random.default_rng(42)
    n_valid = 600
    base_starts = pd.date_range("2023-01-01 00:00:00", "2023-01-12 23:00:00", freq="h")
    chosen_starts = rng.choice(base_starts, size=n_valid)
    durations = pd.to_timedelta(rng.integers(5, 45, size=n_valid), unit="m")

    valid_trips = pd.DataFrame(
        {
            "Viaje_Id": np.arange(1000, 1000 + n_valid),
            "Inicio_del_viaje": pd.Series(chosen_starts).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Fin_del_viaje": pd.Series(chosen_starts + durations).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Origen_Id": rng.choice([51, 52], size=n_valid),
            "Destino_Id": rng.choice([51, 52], size=n_valid),
        }
    )

    # Registros invalidos: formato barra (noviembre), duracion <= 0, anio != 2023 y duplicado
    dirty_rows = pd.DataFrame(
        [
            {
                "Viaje_Id": 9001,
                "Inicio_del_viaje": "15/11/2023 08:30:00",
                "Fin_del_viaje": "15/11/2023 08:45:00",
                "Origen_Id": 51,
                "Destino_Id": 52,
            },
            {
                "Viaje_Id": 9002,
                "Inicio_del_viaje": "2023-01-05 10:00:00",
                "Fin_del_viaje": "2023-01-05 09:50:00",  # fin < inicio (descartar)
                "Origen_Id": 51,
                "Destino_Id": 52,
            },
            {
                "Viaje_Id": 9003,
                "Inicio_del_viaje": "2022-12-31 23:50:00",  # fuera de anio (descartar)
                "Fin_del_viaje": "2023-01-01 00:10:00",
                "Origen_Id": 51,
                "Destino_Id": 52,
            },
        ]
    )
    duplicate_row = valid_trips.iloc[[0]].copy()
    return pd.concat([valid_trips, dirty_rows, duplicate_row], ignore_index=True)


def test_end_to_end_data_features_and_modeling_pipeline(tmp_path):
    raw_df = _build_raw_trips_fixture()

    # 1. Limpieza de viajes crudos
    cleaned, quality = clean_trips(raw_df, year=2023)
    assert quality["metricas"]["filas_limpias"] == 601  # 600 validos + 1 con fecha DD/MM/YYYY
    assert quality["metricas"]["descartados_duracion_no_positiva"] == 1
    assert quality["metricas"]["descartados_fuera_del_anio"] == 1
    assert quality["metricas"]["descartados_duplicados"] == 1

    # 2. Malla horaria continua (2 estaciones x 8760 horas = 17,520 filas)
    grid = build_hourly_grid(cleaned, year=2023)
    assert len(grid) == 2 * 8760
    assert grid["entradas"].sum() == len(cleaned)
    assert grid["salidas"].sum() == len(cleaned)
    validate_hourly_grid(grid)

    # 3. Ingenieria de caracteristicas v2 (29 features explicativas, sin nulos)
    feat_df = add_calendar_and_cyclical_features(grid.copy())
    feat_df = add_historical_lags_and_rolling(feat_df)
    assert len(feat_df) == 2 * (8760 - 168)
    assert set(ALL_FEATURE_COLS).issubset(feat_df.columns)
    assert feat_df[ALL_FEATURE_COLS].isna().sum().sum() == 0

    # 4. Entrenamiento, serializacion e inferencia consolidada
    train_part = feat_df.iloc[:2000]
    eval_part = feat_df.iloc[2000:2300]

    pipe = create_full_pipeline({"max_iter": 15, "max_depth": 4, "random_state": 42})
    pipe.fit(train_part[ALL_FEATURE_COLS], train_part[TARGET_COLS])

    model_file = tmp_path / "pipeline_e2e.joblib"
    joblib.dump(pipe, model_file)
    reloaded = joblib.load(model_file)

    preds = np.clip(reloaded.predict(eval_part[ALL_FEATURE_COLS]), 0, None)
    metrics = evaluate_net_balance(
        eval_part["entradas"].values,
        preds[:, 0],
        eval_part["salidas"].values,
        preds[:, 1],
    )
    actions = classify_rebalancing_action(preds[:, 0] - preds[:, 1])

    assert metrics["balance_mae"] >= 0.0
    assert len(actions) == len(eval_part)
