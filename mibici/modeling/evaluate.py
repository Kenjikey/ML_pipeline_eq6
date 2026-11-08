"""
Evaluation module: Metric calculations, baseline heuristics, and diagnostic metrics
for MiBici MLOps fleet rebalancing.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Computes standard regression metrics: MAE, RMSE, R-squared, and MAPE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip negative predictions to 0 for bike arrival/departure counts
    y_pred_clipped = np.clip(y_pred, 0, None)
    
    mae = float(np.mean(np.abs(y_true - y_pred_clipped)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred_clipped) ** 2)))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred_clipped) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
    
    # Non-zero MAPE to avoid division by zero on quiet station hours
    non_zero_mask = y_true > 0
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred_clipped[non_zero_mask]) / y_true[non_zero_mask])) * 100)
    else:
        mape = 0.0

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae": round(mae, 4),
        f"{p}rmse": round(rmse, 4),
        f"{p}r2": round(r2, 4),
        f"{p}mape": round(mape, 2)
    }

def evaluate_net_balance(
    y_true_in: np.ndarray,
    y_pred_in: np.ndarray,
    y_true_out: np.ndarray,
    y_pred_out: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates net rebalancing demand:
    Balance = Inflow (Entradas) - Outflow (Salidas)
    """
    true_balance = np.asarray(y_true_in) - np.asarray(y_true_out)
    pred_balance = np.asarray(y_pred_in) - np.asarray(y_pred_out)
    
    mae = float(np.mean(np.abs(true_balance - pred_balance)))
    rmse = float(np.sqrt(np.mean((true_balance - pred_balance) ** 2)))
    
    ss_res = np.sum((true_balance - pred_balance) ** 2)
    ss_tot = np.sum((true_balance - np.mean(true_balance)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
    
    return {
        "balance_mae": round(mae, 4),
        "balance_rmse": round(rmse, 4),
        "balance_r2": round(r2, 4)
    }

class BaselineHeuristicModel:
    """
    Historical Median Baseline:
    Predicts demand using the historical median grouped by (estacion_id, dia_semana, hora).
    """
    def __init__(self):
        self.lookup_table: pd.DataFrame = pd.DataFrame()
        self.global_median: float = 0.0

    def fit(self, df: pd.DataFrame, target_col: str = "entradas", station_col: str = "estacion_id"):
        df = df.copy()
        if "hora" not in df.columns and "timestamp" in df.columns:
            df["hora"] = df["timestamp"].dt.hour
        if "dia_semana" not in df.columns and "timestamp" in df.columns:
            df["dia_semana"] = df["timestamp"].dt.dayofweek

        self.global_median = float(df[target_col].median())
        self.lookup_table = (
            df.groupby([station_col, "dia_semana", "hora"])[target_col]
            .median()
            .reset_index(name="baseline_pred")
        )
        return self

    def predict(self, df: pd.DataFrame, station_col: str = "estacion_id") -> np.ndarray:
        df = df.copy()
        if "hora" not in df.columns and "timestamp" in df.columns:
            df["hora"] = df["timestamp"].dt.hour
        if "dia_semana" not in df.columns and "timestamp" in df.columns:
            df["dia_semana"] = df["timestamp"].dt.dayofweek

        merged = df.merge(self.lookup_table, on=[station_col, "dia_semana", "hora"], how="left")
        preds = merged["baseline_pred"].fillna(self.global_median).values
        return preds

if __name__ == "__main__":
    print("Probando mibici/modeling/evaluate.py...")
    y_true = np.array([2, 5, 0, 3, 8])
    y_pred = np.array([2.1, 4.8, 0.2, 3.0, 7.5])
    metrics = calculate_metrics(y_true, y_pred, prefix="test")
    print("Métricas de prueba calculadas:", metrics)