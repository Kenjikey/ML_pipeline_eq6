"""Monitoreo de data drift (PSI/KS) y degradacion de desempeno para MiBici."""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import typer

from mibici.config import FEATURES_FILENAME, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from mibici.modeling.evaluate import evaluate_net_balance
from mibici.pipeline import ALL_FEATURE_COLS

app = typer.Typer()

MONITORED_FEATURES = [
    "entradas_lag_1h",
    "salidas_lag_1h",
    "balance_lag_1h",
    "entradas_roll_mean_3h",
    "salidas_roll_mean_3h",
    "balance_roll_mean_6h",
]


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) entre distribucion base y actual."""
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_dist = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_dist = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    exp_dist = np.where(exp_dist == 0, 1e-6, exp_dist)
    act_dist = np.where(act_dist == 0, 1e-6, act_dist)

    psi = np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))
    return round(float(psi), 4)


def simulate_synthetic_drift(df_ref: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Genera muestra sintetica simulando incremento de demanda en horas pico y ruido de sensor."""
    rng = np.random.default_rng(seed)
    df_synth = df_ref.copy()

    peak_mask = df_synth["es_hora_pico"] == 1
    df_synth.loc[peak_mask, "entradas_lag_1h"] = df_synth.loc[
        peak_mask, "entradas_lag_1h"
    ] * 1.65 + rng.uniform(1.5, 4.0, peak_mask.sum())
    df_synth.loc[peak_mask, "salidas_lag_1h"] = df_synth.loc[
        peak_mask, "salidas_lag_1h"
    ] * 1.40 + rng.uniform(0.5, 2.5, peak_mask.sum())
    df_synth["entradas_roll_mean_3h"] = df_synth["entradas_roll_mean_3h"] * 1.45 + 1.2
    df_synth["salidas_roll_mean_3h"] = df_synth["salidas_roll_mean_3h"] * 1.30 + 0.8
    df_synth["balance_lag_1h"] = df_synth["entradas_lag_1h"] - df_synth["salidas_lag_1h"]

    # Simular cambio de concepto en la demanda real observada
    df_synth["entradas"] = np.round(
        df_synth["entradas"] * 1.35 + rng.normal(1.0, 1.5, len(df_synth)).clip(min=0)
    )
    df_synth["balance"] = df_synth["entradas"] - df_synth["salidas"]
    return df_synth


def evaluate_drift_and_degradation(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame,
    pipeline,
    psi_threshold: float = 0.20,
    mae_degradation_ratio: float = 1.25,
) -> dict:
    """Compara referencia vs muestra actual, mide drift y degradacion y emite alertas."""
    ref_preds = np.clip(pipeline.predict(df_ref[ALL_FEATURE_COLS]), 0, None)
    curr_preds = np.clip(pipeline.predict(df_curr[ALL_FEATURE_COLS]), 0, None)

    ref_metrics = evaluate_net_balance(
        df_ref["entradas"].values, ref_preds[:, 0], df_ref["salidas"].values, ref_preds[:, 1]
    )
    curr_metrics = evaluate_net_balance(
        df_curr["entradas"].values, curr_preds[:, 0], df_curr["salidas"].values, curr_preds[:, 1]
    )

    feature_drift = {}
    alerts = []

    for col in MONITORED_FEATURES:
        psi_val = compute_psi(df_ref[col].values, df_curr[col].values)
        ks_stat, ks_pval = ks_2samp(df_ref[col].values, df_curr[col].values)
        drifted = psi_val >= psi_threshold
        feature_drift[col] = {
            "psi": psi_val,
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "drift_detected": drifted,
        }
        if drifted:
            alerts.append(
                f"DATA_DRIFT: '{col}' excede umbral PSI ({psi_val:.4f} >= {psi_threshold})"
            )

    if curr_metrics["balance_mae"] > ref_metrics["balance_mae"] * mae_degradation_ratio:
        pct_inc = ((curr_metrics["balance_mae"] / ref_metrics["balance_mae"]) - 1.0) * 100
        alerts.append(
            f"MODEL_DEGRADATION: balance_mae subio de {ref_metrics['balance_mae']:.4f} a "
            f"{curr_metrics['balance_mae']:.4f} (+{pct_inc:.1f}%)"
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_sizes": {"reference": len(df_ref), "current": len(df_curr)},
        "reference_metrics": ref_metrics,
        "current_metrics": curr_metrics,
        "feature_drift": feature_drift,
        "status": "ALERT" if alerts else "OK",
        "alerts": alerts,
    }


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / FEATURES_FILENAME,
    model_path: Path = MODELS_DIR / "mibici_balance_pipeline.joblib",
    report_dir: Path = REPORTS_DIR / "monitoring",
    sample_size: int = 25000,
):
    df = pd.read_parquet(features_path)
    df_ref = df[df["mes"] == 10].head(sample_size).copy()
    df_synth = simulate_synthetic_drift(df_ref)

    pipeline = joblib.load(model_path)
    report = evaluate_drift_and_degradation(df_ref, df_synth, pipeline)

    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        report_dir / f"drift_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info(f"Reporte de monitoreo guardado en: {out_file}")
    if report["alerts"]:
        for alert in report["alerts"]:
            logger.warning(alert)
    else:
        logger.success("Sin alertas de drift ni degradacion.")


if __name__ == "__main__":
    app()
