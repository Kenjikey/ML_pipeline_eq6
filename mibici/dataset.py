"""Carga, diagnóstico, limpieza y balance horario de los viajes de MiBici."""

from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mibici.config import (
    INTERIM_DATA_DIR,
    MLFLOW_TRACKING_URI,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TARGET_YEAR,
)
from mibici.tracking import data_run, file_metadata, log_dataframe, save_report

app = typer.Typer()
TRIP_COLUMNS = ["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
CLEANING_RULES = [
    "Excluir viajes sin identificadores enteros positivos o sin fechas interpretables.",
    "Excluir viajes con fin anterior o igual al inicio.",
    "Conservar viajes cuyo inicio y fin pertenecen al año seleccionado.",
    "Eliminar duplicados coincidentes en las cinco columnas necesarias para contar viajes.",
    "Detener el proceso si un Viaje_Id válido describe movimientos distintos.",
    "No eliminar por datos demográficos faltantes: no se usan para contar movimientos.",
    "Reportar duraciones atípicas sin eliminarlas automáticamente.",
    "Balance = entradas - salidas; completar con cero las horas sin movimientos registrados.",
]


def monthly_files(raw_dir: Path, year: int) -> list[Path]:
    """No convertir meses faltantes en meses de demanda cero."""
    expected = [raw_dir / f"datos_abiertos_{year}_{month:02d}.csv" for month in range(1, 13)]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan CSV mensuales para la malla anual: {', '.join(missing)}")
    return expected


def clean_trips(trips: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    """Aplica descartes secuenciales: cada fila eliminada se cuenta una sola vez."""
    missing = set(TRIP_COLUMNS) - set(trips.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    trips = trips.copy()
    metrics = {"filas_entrada": len(trips)}
    nulls = {col: int(value) for col, value in trips.isna().sum().items()}
    for col in ["Viaje_Id", "Origen_Id", "Destino_Id"]:
        trips[col] = pd.to_numeric(trips[col], errors="coerce")
    for col in ["Inicio_del_viaje", "Fin_del_viaje"]:
        trips[col] = pd.to_datetime(trips[col], format="mixed", errors="coerce")

    ids = trips[["Viaje_Id", "Origen_Id", "Destino_Id"]]
    valid_ids = (np.isfinite(ids) & (ids > 0) & (ids % 1 == 0)).all(axis=1)
    filters = [
        ("descartados_identificadores_invalidos", valid_ids),
        (
            "descartados_fechas_invalidas",
            trips[["Inicio_del_viaje", "Fin_del_viaje"]].notna().all(axis=1),
        ),
        ("descartados_duracion_no_positiva", trips["Fin_del_viaje"] > trips["Inicio_del_viaje"]),
        (
            "descartados_fuera_del_anio",
            (trips["Inicio_del_viaje"].dt.year == year) & (trips["Fin_del_viaje"].dt.year == year),
        ),
    ]
    keep = pd.Series(True, index=trips.index)
    for name, valid in filters:
        metrics[name] = int((keep & ~valid).sum())
        keep &= valid
    trips = trips.loc[keep].copy()
    before = len(trips)
    trips = trips.drop_duplicates(subset=TRIP_COLUMNS)
    metrics["descartados_duplicados"] = before - len(trips)
    if trips["Viaje_Id"].duplicated().any():
        raise ValueError("Hay Viaje_Id repetidos con movimientos distintos; revisar el origen.")
    if trips.empty:
        raise ValueError("No quedaron viajes válidos después de la limpieza.")
    for col in ["Viaje_Id", "Origen_Id", "Destino_Id"]:
        trips[col] = trips[col].astype("int64")

    seconds = (trips["Fin_del_viaje"] - trips["Inicio_del_viaje"]).dt.total_seconds()
    q1, q3 = seconds.quantile([0.25, 0.75])
    iqr = q3 - q1
    metrics.update(
        {
            "filas_limpias": len(trips),
            "descartados_total": metrics["filas_entrada"] - len(trips),
            "duracion_atipica_iqr": int(
                ((seconds < q1 - 1.5 * iqr) | (seconds > q3 + 1.5 * iqr)).sum()
            ),
            "viajes_menores_60s_misma_estacion": int(
                ((seconds < 60) & (trips["Origen_Id"] == trips["Destino_Id"])).sum()
            ),
            "viajes_mayores_24h": int((seconds > 86400).sum()),
        }
    )
    return trips.reset_index(drop=True), {
        "metricas": metrics,
        "nulos_entrada": nulls,
        "reglas": CLEANING_RULES,
        "duracion_segundos": {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "min": float(seconds.min()),
            "max": float(seconds.max()),
        },
    }


def load_all_trips(raw_dir: Path, year: int, report: dict | None = None) -> pd.DataFrame:
    frames, inputs = [], []
    for path in tqdm(monthly_files(raw_dir, year), desc="Leyendo CSVs"):
        try:
            frame = pd.read_csv(path, encoding="utf-8", low_memory=False)
            encoding = "utf-8"
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="latin1", low_memory=False)
            encoding = "latin1"
        frame.columns = frame.columns.str.strip()
        if frame.empty:
            raise ValueError(f"El archivo mensual está vacío: {path.name}")
        frames.append(frame)
        inputs.append({**file_metadata(path), "filas": len(frame), "encoding": encoding})
    cleaned, quality = clean_trips(pd.concat(frames, ignore_index=True), year)
    if cleaned["Inicio_del_viaje"].dt.month.nunique() != 12:
        raise ValueError("No hay viajes limpios en cada uno de los 12 meses; revisar cobertura.")
    if report is not None:
        report.update(quality)
        report["entradas"] = inputs
    return cleaned


def build_hourly_grid(trips: pd.DataFrame, year: int) -> pd.DataFrame:
    """Cuenta movimientos; los ceros no prueban disponibilidad de una estación."""
    departures = trips.groupby(["Origen_Id", trips["Inicio_del_viaje"].dt.floor("h")]).size()
    departures.index.names = ["estacion_id", "timestamp"]
    arrivals = trips.groupby(["Destino_Id", trips["Fin_del_viaje"].dt.floor("h")]).size()
    arrivals.index.names = ["estacion_id", "timestamp"]
    stations = np.union1d(trips["Origen_Id"].unique(), trips["Destino_Id"].unique())
    hours = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    index = pd.MultiIndex.from_product([stations, hours], names=["estacion_id", "timestamp"])
    grid = pd.DataFrame(
        {
            "salidas": departures.reindex(index, fill_value=0),
            "entradas": arrivals.reindex(index, fill_value=0),
        }
    ).reset_index()
    grid["balance"] = grid["entradas"] - grid["salidas"]
    if grid["entradas"].sum() != len(trips) or grid["salidas"].sum() != len(trips):
        raise ValueError("La malla no conserva el total de movimientos de los viajes limpios.")
    return grid


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path | None = None,
    year: int = TARGET_YEAR,
    report_dir: Path = REPORTS_DIR / "quality",
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    output_path = output_path or INTERIM_DATA_DIR / f"hourly_balance_{year}.parquet"
    report = {"anio": year}
    with data_run("dataset", tracking_uri) as run_id:
        mlflow.log_params(
            {
                "anio": year,
                "frecuencia": "1h",
                "meses_requeridos": 12,
                "eliminar_atipicos_por_duracion": False,
            }
        )
        trips = load_all_trips(raw_dir, year, report)
        log_dataframe(trips[TRIP_COLUMNS], raw_dir, "viajes_limpios")
        grid = build_hourly_grid(trips, year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(output_path, index=False)
        report["salida"] = file_metadata(output_path)
        report["supuesto_ceros"] = (
            "Cero significa ausencia de movimientos registrados. No se conocen fechas de alta, "
            "baja o fallos de cada estación; no demuestra demanda cero ni disponibilidad de bicis."
        )
        station_month = pd.concat(
            [
                pd.DataFrame(
                    {"estacion": trips["Origen_Id"], "mes": trips["Inicio_del_viaje"].dt.month}
                ),
                pd.DataFrame(
                    {"estacion": trips["Destino_Id"], "mes": trips["Fin_del_viaje"].dt.month}
                ),
            ]
        ).drop_duplicates()
        report["metricas"].update(
            {
                "archivos_entrada": 12,
                "filas_salida": len(grid),
                "columnas_salida": len(grid.columns),
                "estaciones": int(grid["estacion_id"].nunique()),
                "estacion_meses_sin_movimientos": int(
                    grid["estacion_id"].nunique() * 12 - len(station_month)
                ),
                "entradas_total": int(grid["entradas"].sum()),
                "salidas_total": int(grid["salidas"].sum()),
                "horas_sin_movimientos": int(
                    ((grid["entradas"] == 0) & (grid["salidas"] == 0)).sum()
                ),
            }
        )
        log_dataframe(grid, output_path, "salida")
        report_path = save_report(report, report_dir, "dataset", run_id)
    logger.success(f"Balance horario: {output_path}; reporte: {report_path}")


if __name__ == "__main__":
    app()
