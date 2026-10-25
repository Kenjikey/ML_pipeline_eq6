from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import typer
import mlflow
import mlflow.data
import mlflow.sklearn

from mibici.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_YEAR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

app = typer.Typer()

def split_out_of_time(df: pd.DataFrame):
    """
    Particion cronologica estricta para evitar data leakage:
    - Train: Enero a Septiembre (meses 1 a 9)
    - Val: Octubre (mes 10)
    - Test: Noviembre y Diciembre (meses 11 y 12)
    """
    train_df = df[df["mes"] <= 9].copy()
    val_df = df[df["mes"] == 10].copy()
    test_df = df[df["mes"] >= 11].copy()

    logger.info("Particion cronologica completada:")
    logger.info(f" - Train: {len(train_df):,} registros ({train_df['timestamp'].min().date()} a {train_df['timestamp'].max().date()})")
    logger.info(f" - Val:   {len(val_df):,} registros ({val_df['timestamp'].min().date()} a {val_df['timestamp'].max().date()})")
    logger.info(f" - Test:  {len(test_df):,} registros ({test_df['timestamp'].min().date()} a {test_df['timestamp'].max().date()})")

    return train_df, val_df, test_df

def evaluate_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Calcula metricas de Baseline 0 y Baseline Heuristico en el conjunto de prueba."""
    y_test_balance = test_df["balance"].values

    # 1. Baseline 0 (Inercia: prediccion = 0)
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

    logger.info("Metricas de Baselines (Conjunto Test):")
    logger.info(f" - Baseline 0 (Inercia):       MAE = {mae_base0:.4f} bicis | RMSE = {rmse_base0:.4f}")
    logger.info(f" - Baseline Heuristico (Hist): MAE = {mae_heur:.4f} bicis | RMSE = {rmse_heur:.4f}")

    return {
        "mae_base0": mae_base0,
        "rmse_base0": rmse_base0,
        "mae_heur": mae_heur,
        "rmse_heur": rmse_heur,
    }

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
    models_dir: Path = MODELS_DIR,
    max_iter: int = 100,
    max_depth: int = 10,
    learning_rate: float = 0.1,
    random_state: int = 42,
):
    if not features_path.exists():
        logger.error(f"No se encontro el archivo de caracteristicas: {features_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {features_path}")

    logger.info(f"Cargando dataset procesado desde: {features_path}")
    df = pd.read_parquet(features_path)

    train_df, val_df, test_df = split_out_of_time(df)

    # Definir caracteristicas de entrada (excluyendo targets y timestamps)
    exclude_cols = ["timestamp", "entradas", "salidas", "balance"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train, y_train_ent, y_train_sal = train_df[feature_cols], train_df["entradas"], train_df["salidas"]
    X_test, y_test_ent, y_test_sal = test_df[feature_cols], test_df["entradas"], test_df["salidas"]
    y_test_balance = test_df["balance"].values

    # Evaluacion de lineas base
    baseline_metrics = evaluate_baselines(train_df, test_df)

    # Configuracion MLflow
    logger.info(f"Conectando a MLflow Tracking Server en: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="hist_gradient_boosting_decoupled"):
        params = {
            "max_iter": max_iter,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(baseline_metrics)

        # Registro y versionado formal del dataset en MLflow (Requisito 4)
        logger.info("Registrando metadatos y version del dataset en MLflow...")
        dataset_source = mlflow.data.from_pandas(
            df=train_df[feature_cols + ["balance"]],
            source=str(features_path),
            name=f"mibici_features_{TARGET_YEAR}",
            targets="balance",
        )
        mlflow.log_input(dataset_source, context="training")

        # 1. Entrenar Modelo de Entradas
        logger.info("Entrenando modelo de Entradas (HistGradientBoosting)...")
        model_entradas = HistGradientBoostingRegressor(**params)
        model_entradas.fit(X_train, y_train_ent)

        # 2. Entrenar Modelo de Salidas
        logger.info("Entrenando modelo de Salidas (HistGradientBoosting)...")
        model_salidas = HistGradientBoostingRegressor(**params)
        model_salidas.fit(X_train, y_train_sal)

        # 3. Prediccion y calculo de Balance en Test
        pred_ent_test = model_entradas.predict(X_test)
        pred_sal_test = model_salidas.predict(X_test)
        pred_balance_test = pred_ent_test - pred_sal_test

        mae_ent = mean_absolute_error(y_test_ent, pred_ent_test)
        mae_sal = mean_absolute_error(y_test_sal, pred_sal_test)
        mae_balance = mean_absolute_error(y_test_balance, pred_balance_test)
        rmse_balance = np.sqrt(mean_squared_error(y_test_balance, pred_balance_test))
        mejora_vs_heur = ((baseline_metrics["mae_heur"] - mae_balance) / baseline_metrics["mae_heur"]) * 100

        logger.info("Resultados en Test (Nov - Dic 2023):")
        logger.info(f" - MAE Entradas:  {mae_ent:.4f} bicis")
        logger.info(f" - MAE Salidas:   {mae_sal:.4f} bicis")
        logger.info(f" - MAE Balance:   {mae_balance:.4f} bicis")
        logger.info(f" - RMSE Balance:  {rmse_balance:.4f}")
        logger.info(f" - Mejora vs Baseline Heuristico: {mejora_vs_heur:.2f}%")

        mlflow.log_metrics({
            "test_mae_entradas": mae_ent,
            "test_mae_salidas": mae_sal,
            "test_mae_balance": mae_balance,
            "test_rmse_balance": rmse_balance,
            "mejora_vs_heuristico_pct": mejora_vs_heur,
        })

        # Guardar binarios en models/
        model_ent_path = models_dir / "model_entradas.joblib"
        model_sal_path = models_dir / "model_salidas.joblib"
        joblib.dump(model_entradas, model_ent_path)
        joblib.dump(model_salidas, model_sal_path)
        logger.info(f"Modelos respaldados localmente en: {models_dir}")

        # Registrar modelos en MLflow con cloudpickle
        mlflow.sklearn.log_model(
            sk_model=model_entradas,
            artifact_path="model_entradas",
            serialization_format="cloudpickle",
        )
        mlflow.sklearn.log_model(
            sk_model=model_salidas,
            artifact_path="model_salidas",
            serialization_format="cloudpickle",
        )
        logger.success("Modelos, metricas y dataset registrados exitosamente en MLflow.")

if __name__ == "__main__":
    app()