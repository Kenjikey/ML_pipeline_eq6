import pandas as pd
import numpy as np
from pathlib import Path
from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, TARGET_YEAR

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables temporales ciclicas y de calendario."""
    df["hora"] = df["timestamp"].dt.hour
    df["dia_semana"] = df["timestamp"].dt.dayofweek
    df["mes"] = df["timestamp"].dt.month
    
    # Transformaciones ciclicas (seno y coseno)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24.0)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24.0)
    df["dia_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7.0)
    df["dia_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7.0)
    
    # Banderas binarias
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["es_hora_pico"] = (
        ((df["hora"] >= 7) & (df["hora"] <= 9)) |
        ((df["hora"] >= 18) & (df["hora"] <= 20))
    ).astype(int)
    
    return df

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera lags historicos y promedios moviles por estacion."""
    df = df.sort_values(["estacion_id", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("estacion_id")
    
    target_cols = ["entradas", "salidas", "balance"]
    
    for col in target_cols:
        # Lags temporales
        df[f"{col}_lag_1h"] = grouped[col].shift(1)
        df[f"{col}_lag_2h"] = grouped[col].shift(2)
        df[f"{col}_lag_24h"] = grouped[col].shift(24)
        df[f"{col}_lag_168h"] = grouped[col].shift(168)
        
        # Medias moviles de 3 y 6 horas (excluyendo el punto actual t)
        df[f"{col}_roll_mean_3h"] = grouped[col].shift(1).rolling(window=3).mean()
        df[f"{col}_roll_mean_6h"] = grouped[col].shift(1).rolling(window=6).mean()
    
    # Eliminar filas iniciales con NaN generadas por los lags (primeros 7 dias)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def generate_processed_features() -> pd.DataFrame:
    """Pipeline completo de generacion de caracteristicas para modelado."""
    input_path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo intermedio no encontrado: {input_path}")
    
    print(f"Cargando matriz horaria desde: {input_path}")
    df = pd.read_parquet(input_path)
    
    print("Calculando caracteristicas temporales ciclicas...")
    df = add_time_features(df)
    
    print("Calculando lags historicos y promedios moviles...")
    df = add_lag_and_rolling_features(df)
    
    output_path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Dataset procesado guardado en: {output_path}")
    print(f"Dimensiones finales: {df.shape[0]:,} filas x {df.shape[1]} columnas.")
    return df

if __name__ == "__main__":
    generate_processed_features()