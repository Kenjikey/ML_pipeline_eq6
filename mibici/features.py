from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer

from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, TARGET_YEAR

app = typer.Typer()

def add_calendar_and_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables temporales basicas y transformaciones ciclicas."""
    logger.info("Generando variables de calendario y codificacion ciclica...")
    df["mes"] = df["timestamp"].dt.month
    df["dia_mes"] = df["timestamp"].dt.day
    df["dia_semana"] = df["timestamp"].dt.dayofweek
    df["hora"] = df["timestamp"].dt.hour

    # Banderas de calendario
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_hora_pico"] = df["hora"].isin([7, 8, 9, 18, 19, 20]).astype(int)

    # Transformaciones trigonometricas (continuidad horaria y semanal)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    return df

def add_historical_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retardos temporales y medias moviles agrupados por estacion.
    Se asegura el ordenamiento previo para evitar fugas de informacion.
    """
    logger.info("Calculando lags historicos y ventanas moviles por estacion...")
    df = df.sort_values(["estacion_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("estacion_id")

    # Lags clave: t-1h, t-2h, t-24h (mismo horario ayer), t-168h (mismo horario semana previa)
    lags = [1, 2, 24, 168]
    targets = ["entradas", "salidas", "balance"]

    for target in targets:
        for lag in lags:
            df[f"{target}_lag_{lag}h"] = grouped[target].shift(lag)

    # Medias moviles de 3h y 6h (desplazadas 1 periodo para evitar data leakage)
    for target in targets:
        df[f"{target}_roll_mean_3h"] = grouped[target].shift(1).rolling(3).mean()
        df[f"{target}_roll_mean_6h"] = grouped[target].shift(1).rolling(6).mean()

    # Eliminar registros iniciales con nulos debido al lag maximo de 168 horas (7 dias)
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    logger.info(f"Registros iniciales removidos por ventana de inicializacion (168h): {dropped_rows:,}")

    return df

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet",
    output_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
):
    if not input_path.exists():
        logger.error(f"No se encontro el archivo intermedio: {input_path}")
        raise FileNotFoundError(f"Archivo intermedio no encontrado: {input_path}")

    logger.info(f"Cargando dataset intermedio desde: {input_path}")
    df = pd.read_parquet(input_path)

    df = add_calendar_and_cyclical_features(df)
    df = add_historical_lags_and_rolling(df)

    logger.info(f"Guardando dataset procesado con caracteristicas en: {output_path}")
    df.to_parquet(output_path, index=False)
    logger.success("Generacion de caracteristicas completada con exito.")
    logger.info(f"Dimensiones finales: {df.shape[0]:,} filas x {df.shape[1]} columnas.")

if __name__ == "__main__":
    app()