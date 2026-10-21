from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import typer

from mibici.config import FIGURES_DIR, PROCESSED_DATA_DIR, TARGET_YEAR

app = typer.Typer()

def configure_visual_style():
    """Configura el estilo visual para graficos de reporte."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

def plot_hourly_flow(df: pd.DataFrame, figures_dir: Path):
    """Grafico 1: Flujo horario promedio de entradas y salidas."""
    logger.info("Generando grafico de flujo horario promedio...")
    hourly = df.groupby("hora")[["entradas", "salidas"]].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly["hora"], hourly["entradas"], marker="o", label="Entradas (Llegadas)", color="#2b5c8f", linewidth=2)
    plt.plot(hourly["hora"], hourly["salidas"], marker="s", label="Salidas (Viajes iniciados)", color="#d95f02", linewidth=2)
    plt.axvspan(7, 9, color="gray", alpha=0.15, label="Pico Matutino (07:00-09:00)")
    plt.axvspan(18, 20, color="orange", alpha=0.15, label="Pico Vespertino (18:00-20:00)")

    plt.title("Flujo Horario Promedio por Estacion (MiBici 2023)")
    plt.xlabel("Hora del Dia (0 a 23 hrs)")
    plt.ylabel("Bicicletas Promedio por Estacion/Hora")
    plt.xticks(range(0, 24))
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_file = figures_dir / "01_flujo_horario_pico.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Guardado: {out_file}")

def plot_weekday_demand(df: pd.DataFrame, figures_dir: Path):
    """Grafico 2: Demanda acumulada por dia de la semana."""
    logger.info("Generando grafico de demanda por dia de la semana...")
    days_map = {
        0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves",
        4: "Viernes", 5: "Sabado", 6: "Domingo"
    }
    weekday_df = df.copy()
    weekday_df["nombre_dia"] = weekday_df["dia_semana"].map(days_map)
    days_order = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]

    grouped = weekday_df.groupby("nombre_dia")["salidas"].sum().reindex(days_order).reset_index()

    plt.figure(figsize=(9, 4.5))
    colors = ["#2b5c8f" if d not in ["Sabado", "Domingo"] else "#7570b3" for d in days_order]
    sns.barplot(data=grouped, x="nombre_dia", y="salidas", palette=colors)

    plt.title("Volumen Total de Viajes por Dia de la Semana (MiBici 2023)")
    plt.xlabel("Dia de la Semana")
    plt.ylabel("Total de Salidas Registradas")
    plt.tight_layout()

    out_file = figures_dir / "02_demanda_dia_semana.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Guardado: {out_file}")

def plot_station_imbalance(df: pd.DataFrame, figures_dir: Path):
    """Grafico 3: Top estaciones con mayor superavit y deficit anual."""
    logger.info("Generando grafico de estaciones con mayor desbalance...")
    station_balance = df.groupby("estacion_id")["balance"].sum().reset_index()

    top_superavit = station_balance.sort_values("balance", ascending=False).head(8)
    top_deficit = station_balance.sort_values("balance", ascending=True).head(8)
    combined = pd.concat([top_deficit, top_superavit]).sort_values("balance")
    combined["estacion_label"] = combined["estacion_id"].astype(str).radd("Estacion ")

    plt.figure(figsize=(10, 6))
    colors = ["#d95f02" if b < 0 else "#1b9e77" for b in combined["balance"]]
    plt.barh(combined["estacion_label"], combined["balance"], color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

    plt.title("Estaciones Criticas de Rebalanceo: Mayor Deficit vs Superavit (2023)")
    plt.xlabel("Balance Neto Anual Acumulado (Entradas - Salidas)")
    plt.ylabel("Estacion")
    plt.tight_layout()

    out_file = figures_dir / "03_top_estaciones_desbalance.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Guardado: {out_file}")

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
    figures_dir: Path = FIGURES_DIR,
):
    if not input_path.exists():
        logger.error(f"Dataset de caracteristicas no encontrado: {input_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_visual_style()

    logger.info(f"Cargando datos para generacion de graficos desde: {input_path}")
    df = pd.read_parquet(input_path)

    plot_hourly_flow(df, figures_dir)
    plot_weekday_demand(df, figures_dir)
    plot_station_imbalance(df, figures_dir)

    logger.success(f"Generacion de graficos completada. Archivos disponibles en: {figures_dir}")

if __name__ == "__main__":
    app()