"""
Scikit-Learn Pipeline integration: Standardized ColumnTransformers, custom feature transformers,
and reproducible model training wrappers for MiBici MLOps.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

class CyclicalFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to encode cyclical date/time attributes (hora, dia_semana).
    """
    def __init__(self, hour_col: Optional[str] = "hora", dayofweek_col: Optional[str] = "dia_semana"):
        self.hour_col = hour_col
        self.dayofweek_col = dayofweek_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "timestamp" in X.columns and not pd.api.types.is_datetime64_any_dtype(X["timestamp"]):
            X["timestamp"] = pd.to_datetime(X["timestamp"])

        if "timestamp" in X.columns:
            hora = X["timestamp"].dt.hour
            dia_semana = X["timestamp"].dt.dayofweek
        else:
            hora = X.get(self.hour_col, 0)
            dia_semana = X.get(self.dayofweek_col, 0)

        X["hora_sin"] = np.sin(2 * np.pi * hora / 24.0)
        X["hora_cos"] = np.cos(2 * np.pi * hora / 24.0)
        X["dia_semana_sin"] = np.sin(2 * np.pi * dia_semana / 7.0)
        X["dia_semana_cos"] = np.cos(2 * np.pi * dia_semana / 7.0)
        return X

def get_feature_names(df_columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Returns categorized feature names used across models."""
    if df_columns is not None:
        numeric_cols = [c for c in df_columns if "lag" in c or "roll" in c or "media" in c or "std" in c]
        cyclical_cols = [c for c in df_columns if c in ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]]
        categorical_cols = [c for c in df_columns if c in ["es_fin_de_semana", "es_hora_pico", "dia_semana", "mes", "hora"]]
    else:
        numeric_cols = [f"{t}_lag_{lag}" for t in ["inflow", "outflow"] for lag in [1, 2, 3, 24, 168]]
        cyclical_cols = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]
        categorical_cols = ["es_fin_de_semana", "es_hora_pico"]

    return {
        "numeric_features": numeric_cols,
        "cyclical_features": cyclical_cols,
        "categorical_features": categorical_cols,
        "all_model_features": numeric_cols + cyclical_cols + categorical_cols
    }

def build_preprocessing_pipeline(numeric_cols: Optional[List[str]] = None) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer to scale numeric lags/rolling stats and pass through cyclical signals.
    """
    if numeric_cols is None:
        feature_dict = get_feature_names()
        numeric_cols = feature_dict["numeric_features"]
        cyc_cols = feature_dict["cyclical_features"]
        cat_cols = feature_dict["categorical_features"]
    else:
        cyc_cols = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos"]
        cat_cols = ["es_fin_de_semana", "es_hora_pico"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_scaler", RobustScaler(), numeric_cols),
            ("cyclical_passthrough", "passthrough", cyc_cols),
            ("cat_passthrough", "passthrough", cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def create_full_pipeline(estimator: Any, numeric_cols: Optional[List[str]] = None) -> Pipeline:
    """Combines the preprocessor and estimator into an end-to-end reproducible Scikit-Learn Pipeline."""
    preprocessor = build_preprocessing_pipeline(numeric_cols)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])

if __name__ == "__main__":
    print("Probando mibici/pipeline.py...")
    pipe = build_preprocessing_pipeline()
    print("Pipeline de preprocesamiento construido exitosamente:", pipe)