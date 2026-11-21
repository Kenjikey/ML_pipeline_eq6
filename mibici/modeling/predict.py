from pathlib import Path
from typing import Optional

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import FEATURES_FILENAME, MODELS_DIR, PROCESSED_DATA_DIR, TARGET_YEAR
from mibici.pipeline import ALL_FEATURE_COLS

app = typer.Typer()


def load_pipeline(models_dir: Path):
    pipeline_path = models_dir / "mibici_balance_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline no encontrado en {pipeline_path}. Ejecuta mibici/modeling/train.py"
        )
    return joblib.load(pipeline_path)


def classify_rebalancing_action(balance: np.ndarray) -> np.ndarray:
    conditions = [balance >= 2.0, balance <= -2.0]
    choices = ["ALERTA SUPERAVIT (Retirar Bicis)", "ALERTA DEFICIT (Abastecer Bicis)"]
    return np.select(conditions, choices, default="ESTABLE")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / FEATURES_FILENAME,
    models_dir: Path = MODELS_DIR,
    predictions_path: Path = PROCESSED_DATA_DIR / f"test_predictions_{TARGET_YEAR}.csv",
    estacion_id: Optional[int] = typer.Option(None, "--estacion", "-e"),
):
    df = pd.read_parquet(features_path)
    if estacion_id is not None:
        df = df[df["estacion_id"] == estacion_id].copy()
    else:
        df = df[df["mes"] >= 11].copy()

    pipeline = load_pipeline(models_dir)
    preds = np.clip(pipeline.predict(df[ALL_FEATURE_COLS]), 0, None)
    pred_ent, pred_sal = preds[:, 0], preds[:, 1]
    pred_balance = pred_ent - pred_sal

    results = pd.DataFrame(
        {
            "estacion_id": df["estacion_id"].values,
            "timestamp": df["timestamp"].values,
            "pred_entradas": np.round(pred_ent, 2),
            "pred_salidas": np.round(pred_sal, 2),
            "pred_balance": np.round(pred_balance, 2),
            "accion_operativa": classify_rebalancing_action(pred_balance),
        }
    )
    results.to_csv(predictions_path, index=False)
    logger.success(f"Inferencia guardada en {predictions_path} ({len(results):,} registros)")


if __name__ == "__main__":
    app()
