from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from mibici.config import INTERIM_DATA_DIR, RAW_DATA_DIR, TARGET_YEAR

app = typer.Typer()

def load_all_trips(raw_dir: Path, year: int) -> pd.DataFrame:
    """Carga y concatena los archivos CSV mensuales de viajes con encoding latin1."""
    csv_files = sorted(raw_dir.glob(f"datos_abiertos_{year}_*.csv"))
    if not csv_files:
        logger.error(f"No se encontraron archivos datos_abiertos_{year}_*.csv en {raw_dir}")
        raise FileNotFoundError(f"No se encontraron archivos en {raw_dir}")

    logger.info(f"Cargando {len(csv_files)} archivos CSV mensuales del anio {year}...")
    df_list = []
    for f in tqdm(csv_files, desc="Leyendo CSVs"):
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df_list.append(df)

    trips = pd.concat(df_list, ignore_index=True)

    # Conversion de marcas de tiempo y limpieza
    logger.info("Convirtiendo marcas de tiempo y limpiando nulos...")
    trips["Inicio_del_viaje"] = pd.to_datetime(trips["Inicio_del_viaje"], errors="coerce")
    trips["Fin_del_viaje"] = pd.to_datetime(trips["Fin_del_viaje"], errors="coerce")

    trips = trips.dropna(subset=["Origen_Id", "Destino_Id", "Inicio_del_viaje", "Fin_del_viaje"])
    trips["Origen_Id"] = trips["Origen_Id"].astype(int)
    trips["Destino_Id"] = trips["Destino_Id"].astype(int)

    # Filtrar estrictamente al anio objetivo
    trips = trips[
        (trips["Inicio_del_viaje"].dt.year == year) & (trips["Fin_del_viaje"].dt.year == year)
    ].reset_index(drop=True)

    logger.info(f"Total de viajes limpios cargados: {len(trips):,}")
    return trips

def build_hourly_grid(trips: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Construye la malla continua horaria por estacion:
    - Salidas (S) en el intervalo horario de inicio de viaje.
    - Entradas (E) en el intervalo horario de fin de viaje.
    - Balance neto = Entradas - Salidas.
    """
    logger.info("Generando malla temporal horaria continua...")
    trips["hora_inicio"] = trips["Inicio_del_viaje"].dt.floor("h")
    trips["hora_fin"] = trips["Fin_del_viaje"].dt.floor("h")

    # Contabilizar salidas
    salidas = (
        trips.groupby(["Origen_Id", "hora_inicio"])
        .size()
        .reset_index(name="salidas")
        .rename(columns={"Origen_Id": "estacion_id", "hora_inicio": "timestamp"})
    )

    # Contabilizar entradas
    entradas = (
        trips.groupby(["Destino_Id", "hora_fin"])
        .size()
        .reset_index(name="entradas")
        .rename(columns={"Destino_Id": "estacion_id", "hora_fin": "timestamp"})
    )

    # Malla cartesiana: todas las estaciones x 8,760 horas del anio
    all_stations = np.union1d(trips["Origen_Id"].unique(), trips["Destino_Id"].unique())
    all_hours = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq="h")

    grid = pd.MultiIndex.from_product(
        [all_stations, all_hours], names=["estacion_id", "timestamp"]
    ).to_frame().reset_index(drop=True)

    # Unir flujos e imputar ceros
    grid = grid.merge(salidas, on=["estacion_id", "timestamp"], how="left")
    grid = grid.merge(entradas, on=["estacion_id", "timestamp"], how="left")
    grid["salidas"] = grid["salidas"].fillna(0).astype(int)
    grid["entradas"] = grid["entradas"].fillna(0).astype(int)
    grid["balance"] = grid["entradas"] - grid["salidas"]

    grid = grid.sort_values(["estacion_id", "timestamp"]).reset_index(drop=True)
    return grid

@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet",
    year: int = TARGET_YEAR,
):
    logger.info(f"Iniciando procesamiento de datos crudos para el anio {year}...")
    trips_df = load_all_trips(raw_dir, year)
    grid_df = build_hourly_grid(trips_df, year)

    grid_df.to_parquet(output_path, index=False)
    logger.success(f"Matriz horaria guardada en: {output_path}")
    logger.info(f"Dimensiones finales: {grid_df.shape[0]:,} filas x {grid_df.shape[1]} columnas.")

if __name__ == "__main__":
    app()