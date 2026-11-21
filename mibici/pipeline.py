"""
Pipeline consolidado de preprocesamiento y modelado multi-salida para MiBici.
Alineado con las 29 variables explicativas generadas en mibici/features.py (v2).
"""

from typing import Any, Dict, List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

TARGET_COLS = ["entradas", "salidas"]
EXCLUDE_COLS = ["timestamp", "entradas", "salidas", "balance"]

STATION_COL = ["estacion_id"]
CALENDAR_COLS = ["mes", "dia_mes", "dia_semana", "hora", "es_fin_de_semana", "es_hora_pico"]
CYCLICAL_COLS = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]
LAG_COLS = [
    f"{target}_lag_{lag}h"
    for target in ["entradas", "salidas", "balance"]
    for lag in [1, 2, 24, 168]
]
ROLLING_COLS = [
    f"{target}_roll_mean_{win}h" for target in ["entradas", "salidas", "balance"] for win in [3, 6]
]

NUMERIC_HISTORY_COLS = LAG_COLS + ROLLING_COLS
ALL_FEATURE_COLS = STATION_COL + CALENDAR_COLS + CYCLICAL_COLS + NUMERIC_HISTORY_COLS


def get_feature_groups() -> Dict[str, List[str]]:
    """Retorna el contrato exacto de columnas de entrada por grupo funcional."""
    return {
        "station_features": STATION_COL,
        "calendar_features": CALENDAR_COLS,
        "cyclical_features": CYCLICAL_COLS,
        "numeric_history_features": NUMERIC_HISTORY_COLS,
        "all_features": ALL_FEATURE_COLS,
        "targets": TARGET_COLS,
    }


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Construye el ColumnTransformer determinista sobre las 29 features de v2."""
    return ColumnTransformer(
        transformers=[
            (
                "station_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                STATION_COL,
            ),
            ("history_scaler", RobustScaler(), NUMERIC_HISTORY_COLS),
            ("calendar_passthrough", "passthrough", CALENDAR_COLS),
            ("cyclical_passthrough", "passthrough", CYCLICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_full_pipeline(params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Fija el pipeline completo: ColumnTransformer -> MultiOutput HistGradientBoosting."""
    default_params: Dict[str, Any] = {
        "loss": "poisson",
        "max_iter": 450,
        "max_depth": None,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 30,
        "learning_rate": 0.05,
        "l2_regularization": 0.1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    base_estimator = HistGradientBoostingRegressor(**default_params)
    multi_estimator = MultiOutputRegressor(base_estimator)

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("estimator", multi_estimator),
        ]
    )
