import pandas as pd
import numpy as np
from pathlib import Path
from mibici.config import RAW_DATA_DIR, INTERIM_DATA_DIR, TARGET_YEAR

def load_all_trips(raw_dir: Path) -> pd.DataFrame:
    """Carga y concatena los 12 archivos CSV mensuales de viajes de 2023."""
    csv_files = sorted(raw_dir.glob("datos_abiertos_2023_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos datos_abiertos_2023_*.csv en {raw_dir}")
    
    print(f"Cargando {len(csv_files)} archivos CSV mensuales del anio {TARGET_YEAR}...")
    df_list = []
    for f in csv_files:
        print(f"Leyendo archivo: {f.name}")
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df_list.append(df)
    
    trips = pd.concat(df_list, ignore_index=True)
    
    # Conversion explicita a datetime
    print("Convirtiendo marcas de tiempo a formato datetime...")
    trips["Inicio_del_viaje"] = pd.to_datetime(trips["Inicio_del_viaje"], errors="coerce")
    trips["Fin_del_viaje"] = pd.to_datetime(trips["Fin_del_viaje"], errors="coerce")
    
    # Limpieza de registros con nulos
    trips = trips.dropna(subset=["Origen_Id", "Destino_Id", "Inicio_del_viaje", "Fin_del_viaje"])
    trips["Origen_Id"] = trips["Origen_Id"].astype(int)
    trips["Destino_Id"] = trips["Destino_Id"].astype(int)
    
    # Filtrar estrictamente al anio objetivo
    trips = trips[
        (trips["Inicio_del_viaje"].dt.year == TARGET_YEAR) & 
        (trips["Fin_del_viaje"].dt.year == TARGET_YEAR)
    ].reset_index(drop=True)
    
    print(f"Total de viajes limpios cargados: {len(trips):,}")
    return trips

def build_hourly_grid(trips: pd.DataFrame, year: int = TARGET_YEAR) -> pd.DataFrame:
    """
    Construye una malla temporal horaria continua por estacion:
    - Salidas (S) asignadas a la hora de inicio del viaje.
    - Entradas (E) asignadas a la hora de fin del viaje.
    - Balance neto = Entradas - Salidas.
    """
    print("Generando malla temporal horaria continua...")
    trips["hora_inicio"] = trips["Inicio_del_viaje"].dt.floor("h")
    trips["hora_fin"] = trips["Fin_del_viaje"].dt.floor("h")

    # Contabilizar salidas por estacion y hora
    salidas = (
        trips.groupby(["Origen_Id", "hora_inicio"])
        .size()
        .reset_index(name="salidas")
        .rename(columns={"Origen_Id": "estacion_id", "hora_inicio": "timestamp"})
    )

    # Contabilizar entradas por estacion y hora
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

    # Unir flujos y rellenar horas sin viajes con cero
    grid = grid.merge(salidas, on=["estacion_id", "timestamp"], how="left")
    grid = grid.merge(entradas, on=["estacion_id", "timestamp"], how="left")
    grid["salidas"] = grid["salidas"].fillna(0).astype(int)
    grid["entradas"] = grid["entradas"].fillna(0).astype(int)
    
    # Calcular balance neto
    grid["balance"] = grid["entradas"] - grid["salidas"]

    grid = grid.sort_values(["estacion_id", "timestamp"]).reset_index(drop=True)
    return grid

if __name__ == "__main__":
    trips_df = load_all_trips(RAW_DATA_DIR)
    grid_df = build_hourly_grid(trips_df, TARGET_YEAR)
    
    output_path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet"
    grid_df.to_parquet(output_path, index=False)
    print(f"Matriz horaria guardada en: {output_path}")
    print(f"Dimensiones finales: {grid_df.shape[0]:,} filas x {grid_df.shape[1]} columnas.")