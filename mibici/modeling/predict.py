from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from mibici.config import MODELS_DIR, PROCESSED_DATA_DIR, TARGET_YEAR

app = typer.Typer()

def load_models(models_dir: Path):
    """Carga los modelos entrenados de Entradas y Salidas."""
    model_ent_path = models_dir / "model_entradas.joblib"
    model_sal_path = models_dir / "model_salidas.joblib"

    if not model_ent_path.exists() or not model_sal_path.exists():
        logger.error(f"Modelos no encontrados en {models_dir}. Ejecuta primero mibici/modeling/train.py")
        raise FileNotFoundError(f"Modelos no encontrados en {models_dir}")

    logger.info(f"Cargando modelos desde: {models_dir}")
    model_entradas = joblib.load(model_ent_path)
    model_salidas = joblib.load(model_sal_path)
    return model_entradas, model_salidas

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
    models_dir: Path = MODELS_DIR,
    predictions_path: Path = PROCESSED_DATA_DIR / f"test_predictions_{TARGET_YEAR}.csv",
    estacion_id: Optional[int] = typer.Option(None, "--estacion", "-e", help="ID opcional de estacion para filtrar"),
):
    if not features_path.exists():
        logger.error(f"Dataset de caracteristicas no encontrado: {features_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {features_path}")

    logger.info(f"Cargando datos para inferencia desde: {features_path}")
    df = pd.read_parquet(features_path)

    # Filtrado opcional por estacion o evaluacion del periodo de prueba (Nov-Dic)
    if estacion_id is not None:
        df = df[df["estacion_id"] == estacion_id].copy()
        logger.info(f"Filtrando inferencia para la estacion: {estacion_id}")
    else:
        # Por defecto evaluar el conjunto de Test (meses 11 y 12)
        df = df[df["mes"] >= 11].copy()
        logger.info(f"Generando inferencia sobre conjunto Test: {len(df):,} registros")

    model_entradas, model_salidas = load_models(models_dir)

    exclude_cols = ["timestamp", "entradas", "salidas", "balance"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    logger.info("Ejecutando inferencia sobre caracteristicas...")
    pred_ent = np.maximum(0, model_entradas.predict(X))
    pred_sal = np.maximum(0, model_salidas.predict(X))
    pred_balance = pred_ent - pred_sal

    results = pd.DataFrame(
        {
            "estacion_id": df["estacion_id"].values,
            "timestamp": df["timestamp"].values,
            "pred_entradas": np.round(pred_ent, 2),
            "pred_salidas": np.round(pred_sal, 2),
            "pred_balance": np.round(pred_balance, 2),
        }
    )

    # Clasificacion de accion logistica para rebalanceo
    conditions = [
        results["pred_balance"] >= 2.0,
        results["pred_balance"] <= -2.0,
    ]
    choices = [
        "ALERTA SUPERAVIT (Retirar Bicis)",
        "ALERTA DEFICIT (Abastecer Bicis)",
    ]
    results["accion_operativa"] = np.select(conditions, choices, default="ESTABLE")

    logger.info(f"Guardando resultados de inferencia en: {predictions_path}")
    results.to_csv(predictions_path, index=False)
    logger.success("Inferencia completada exitosamente.")

    # Muestra de las primeras 5 predicciones
    print("Primeras estimaciones generadas:")
    print(results.head())

if __name__ == "__main__":
    app()