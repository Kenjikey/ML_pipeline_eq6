from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    INTERIM_DATA_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_YEAR,
)
from mibici.tracking import data_run, file_metadata, log_dataframe, save_report

app = typer.Typer()

LAGS = [1, 2, 24, 168]
WINDOWS = [3, 6]
TARGETS = ["entradas", "salidas", "balance"]


def validate_hourly_grid(df: pd.DataFrame) -> None:
    """Los desplazamientos por filas solo equivalen a horas en una serie continua."""
    required = ["estacion_id", "timestamp", *TARGETS]
    if not set(required).issubset(df.columns):
        raise ValueError(f"La malla requiere las columnas {required}")
    if df.empty or df[required].isna().any().any():
        raise ValueError("La malla está vacía o contiene nulos en columnas necesarias.")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("timestamp debe tener tipo datetime.")
    if df.duplicated(["estacion_id", "timestamp"]).any():
        raise ValueError("Hay estaciones y horas duplicadas.")
    if not df["timestamp"].eq(df["timestamp"].dt.floor("h")).all():
        raise ValueError("Los timestamps deben corresponder al inicio de cada hora.")
    counts = df[["entradas", "salidas"]]
    if not (np.isfinite(counts) & (counts >= 0) & (counts % 1 == 0)).all().all():
        raise ValueError("Entradas y salidas deben ser conteos enteros no negativos.")
    if not df["balance"].eq(df["entradas"] - df["salidas"]).all():
        raise ValueError("El balance debe ser entradas menos salidas.")
    ordered = df.sort_values(["estacion_id", "timestamp"])
    delta = ordered.groupby("estacion_id")["timestamp"].diff().dropna()
    if not delta.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Hay huecos temporales: los lags por filas no representarían horas.")


def add_calendar_and_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables temporales basicas y transformaciones ciclicas."""
    logger.info("Generando variables de calendario y codificacion ciclica...")
    df["mes"] = df["timestamp"].dt.month
    df["dia_mes"] = df["timestamp"].dt.day
    df["dia_semana"] = df["timestamp"].dt.dayofweek
    df["hora"] = df["timestamp"].dt.hour

    # Hora pico es una hipótesis explícita del flujo original; no se infiere del test.
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
    history_columns = []
    for target in TARGETS:
        for lag in LAGS:
            df[f"{target}_lag_{lag}h"] = grouped[target].shift(lag)
            history_columns.append(f"{target}_lag_{lag}h")

    # Medias moviles de 3h y 6h (desplazadas 1 periodo para evitar data leakage)
    for target in TARGETS:
        for window in WINDOWS:
            column = f"{target}_roll_mean_{window}h"
            df[column] = grouped[target].transform(
                lambda values: values.shift(1).rolling(window).mean()
            )
            history_columns.append(column)

    # Eliminar registros iniciales con nulos debido al lag maximo de 168 horas (7 dias)
    initial_rows = len(df)
    df = df.dropna(subset=history_columns).reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    logger.info(
        f"Registros iniciales removidos por ventana de inicializacion (168h): {dropped_rows:,}"
    )

    return df


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet",
    output_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
    report_dir: Path = REPORTS_DIR / "quality",
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    if not input_path.exists():
        logger.error(f"No se encontro el archivo intermedio: {input_path}")
        raise FileNotFoundError(f"Archivo intermedio no encontrado: {input_path}")

    with data_run("features", tracking_uri) as run_id:
        mlflow.log_params(
            {
                "lags_horas": str(LAGS),
                "ventanas_horas": str(WINDOWS),
                "desplazamiento_rolling": 1,
                "horas_pico": "7,8,9,18,19,20",
            }
        )
        logger.info(f"Cargando dataset intermedio desde: {input_path}")
        df = pd.read_parquet(input_path)
        validate_hourly_grid(df)
        report = {
            "entradas": [file_metadata(input_path)],
            "metricas": {"filas_entrada": len(df)},
            "reglas": [
                "Validar claves únicas, continuidad horaria y balance = entradas - salidas.",
                "Calendario y seno/coseno de hora y día de la semana.",
                "Retardos de 1, 2, 24 y 168 horas por estación.",
                "Medias de las 3 y 6 horas anteriores por estación; excluir la hora actual.",
                "Eliminar solo filas sin historial suficiente para estas variables.",
                "Conservar unidades de conteo; sin escalado ni PCA para facilitar interpretación.",
            ],
        }
        log_dataframe(df, input_path, "entrada")
        df = add_calendar_and_cyclical_features(df)
        df = add_historical_lags_and_rolling(df)
        if df.empty:
            raise ValueError("No hay historial suficiente para generar las características.")
        if not np.isfinite(df.select_dtypes(include="number")).all().all():
            raise ValueError("Las características contienen valores no finitos.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        report["salida"] = file_metadata(output_path)
        report["intervalo_salida"] = {
            "inicio": df["timestamp"].min().isoformat(),
            "fin": df["timestamp"].max().isoformat(),
        }
        report["metricas"].update(
            {
                "filas_salida": len(df),
                "columnas_salida": len(df.columns),
                "filas_sin_historial": report["metricas"]["filas_entrada"] - len(df),
                "nulos_salida": int(df.isna().sum().sum()),
                "estaciones": int(df["estacion_id"].nunique()),
            }
        )
        log_dataframe(df, output_path, "salida")
        report_path = save_report(report, report_dir, "features", run_id)
    logger.success(f"Características: {output_path}; reporte: {report_path}")


if __name__ == "__main__":
    app()
