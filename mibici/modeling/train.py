import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

from mibici.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_YEAR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def load_processed_data() -> pd.DataFrame:
    """Carga el dataset procesado con caracteristicas."""
    data_path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Archivo procesado no encontrado: {data_path}")
    df = pd.read_parquet(data_path)
    return df

def split_out_of_time(df: pd.DataFrame):
    """
    Partición cronológica estricta para evitar data leakage (fuga de datos):
    - Train: Enero a Septiembre (meses 1-9) -> Datos para entrenamiento.
    - Val: Octubre (mes 10) -> Datos para ajuste de hiperparámetros.
    - Test: Noviembre y Diciembre (meses 11-12) -> Evaluación final.
    """
    train_df = df[df["mes"] <= 9].copy()
    val_df = df[df["mes"] == 10].copy()
    test_df = df[df["mes"] >= 11].copy()
    
    print("Particion completada:")
    print(f" - Train: {len(train_df):,} registros ({train_df['timestamp'].min().date()} a {train_df['timestamp'].max().date()})")
    print(f" - Val:   {len(val_df):,} registros ({val_df['timestamp'].min().date()} a {val_df['timestamp'].max().date()})")
    print(f" - Test:  {len(test_df):,} registros ({test_df['timestamp'].min().date()} a {test_df['timestamp'].max().date()})")
    
    return train_df, val_df, test_df

def evaluate_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Calcula métricas de referencia para comparar el rendimiento del modelo ML:
    - Baseline 0: Predice balance constante en cero (métrica de inercia).
    - Baseline Heurístico: Promedio histórico basado en estación, día y hora.
    """
    y_test_balance = test_df["balance"].values
    
    # 1. Baseline 0 (Balance Cero)
    y_pred_base0 = np.zeros_like(y_test_balance)
    mae_base0 = mean_absolute_error(y_test_balance, y_pred_base0)
    rmse_base0 = np.sqrt(mean_squared_error(y_test_balance, y_pred_base0))
    
    # 2. Baseline Heuristico (Promedio historico por estacion, dia_semana y hora)
    historical_avg = train_df.groupby(["estacion_id", "dia_semana", "hora"])["balance"].mean().reset_index()
    historical_avg.rename(columns={"balance": "pred_balance_heuristico"}, inplace=True)
    
    test_merged = test_df.merge(historical_avg, on=["estacion_id", "dia_semana", "hora"], how="left")
    y_pred_heur = test_merged["pred_balance_heuristico"].fillna(0).values
    
    mae_heur = mean_absolute_error(y_test_balance, y_pred_heur)
    rmse_heur = np.sqrt(mean_squared_error(y_test_balance, y_pred_heur))
    
    print("\nResultados de Lineas Base (Test):")
    print(f" - Baseline 0 (Inercia):       MAE = {mae_base0:.4f} bicis | RMSE = {rmse_base0:.4f}")
    print(f" - Baseline Heuristico (Hist): MAE = {mae_heur:.4f} bicis | RMSE = {rmse_heur:.4f}")
    
    return {
        "mae_base0": mae_base0,
        "rmse_base0": rmse_base0,
        "mae_heur": mae_heur,
        "rmse_heur": rmse_heur
    }

def train_and_evaluate_models():
    """Entrenamiento de modelos desacoplados (Entradas y Salidas) con MLflow Tracking."""
    df = load_processed_data()
    train_df, val_df, test_df = split_out_of_time(df)
    
    # Definir caracteristicas de entrada (excluyendo targets y timestamps)
    exclude_cols = ["timestamp", "entradas", "salidas", "balance"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train, y_train_ent, y_train_sal = train_df[feature_cols], train_df["entradas"], train_df["salidas"]
    X_val, y_val_ent, y_val_sal = val_df[feature_cols], val_df["entradas"], val_df["salidas"]
    X_test, y_test_ent, y_test_sal = test_df[feature_cols], test_df["entradas"], test_df["salidas"]
    y_test_balance = test_df["balance"].values
    
    # Evaluar Baselines
    baseline_metrics = evaluate_baselines(train_df, test_df)
    
    # Configuracion MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="hist_gradient_boosting_decoupled"):
        # Parametros del modelo
        params = {
            "max_iter": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "random_state": 42
        }
        mlflow.log_params(params)
        mlflow.log_metrics(baseline_metrics)
        
        # 1. Entrenar Modelo de Entradas
        print("\nEntrenando modelo de Entradas (HistGradientBoosting)...")
        model_entradas = HistGradientBoostingRegressor(**params)
        model_entradas.fit(X_train, y_train_ent)
        
        # 2. Entrenar Modelo de Salidas
        print("Entrenando modelo de Salidas (HistGradientBoosting)...")
        model_salidas = HistGradientBoostingRegressor(**params)
        model_salidas.fit(X_train, y_train_sal)
        
        # 3. Predicciones en Test y cálculo de balance final
        # El balance se deriva de la diferencia entre entradas y salidas predichas.
        pred_ent_test = model_entradas.predict(X_test)
        pred_sal_test = model_salidas.predict(X_test)
        pred_balance_test = pred_ent_test - pred_sal_test
        
        # 4. Calcular Metricas del Modelo
        mae_ent = mean_absolute_error(y_test_ent, pred_ent_test)
        mae_sal = mean_absolute_error(y_test_sal, pred_sal_test)
        mae_balance = mean_absolute_error(y_test_balance, pred_balance_test)
        rmse_balance = np.sqrt(mean_squared_error(y_test_balance, pred_balance_test))
        
        mejora_vs_heur = ((baseline_metrics["mae_heur"] - mae_balance) / baseline_metrics["mae_heur"]) * 100
        
        print("Resultados del Modelo ML en Test (Nov - Dic 2023):")
        print(f" - MAE Entradas:  {mae_ent:.4f} bicis")
        print(f" - MAE Salidas:   {mae_sal:.4f} bicis")
        print(f" - MAE Balance:   {mae_balance:.4f} bicis")
        print(f" - RMSE Balance:  {rmse_balance:.4f}")
        print(f" - Mejora vs Baseline Heuristico: {mejora_vs_heur:.2f}%")
        
        # Registrar en MLflow
        mlflow.log_metrics({
            "test_mae_entradas": mae_ent,
            "test_mae_salidas": mae_sal,
            "test_mae_balance": mae_balance,
            "test_rmse_balance": rmse_balance,
            "mejora_vs_heuristico_pct": mejora_vs_heur
        })
        
        # Guardar copias locales en models/
        joblib.dump(model_entradas, MODELS_DIR / "model_entradas.joblib")
        joblib.dump(model_salidas, MODELS_DIR / "model_salidas.joblib")
        
        # Registrar modelos en MLflow usando cloudpickle
        mlflow.sklearn.log_model(
            sk_model=model_entradas,
            artifact_path="model_entradas",
            serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            sk_model=model_salidas,
            artifact_path="model_salidas",
            serialization_format="cloudpickle"
        )
        
        print("\nExperimento y modelos registrados exitosamente en MLflow y guardados en models/.")

if __name__ == "__main__":
    train_and_evaluate_models()