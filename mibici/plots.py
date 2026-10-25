"""EDA reproducible: tablas, figuras e interpretaciones del balance de MiBici."""

import json
from pathlib import Path

from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import typer

from mibici.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_YEAR,
)
from mibici.features import validate_hourly_grid
from mibici.tracking import data_run, file_metadata, log_dataframe, save_report

app = typer.Typer()
matplotlib.use("Agg")
DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
CORR_COLUMNS = [
    "entradas",
    "salidas",
    "balance",
    "entradas_lag_1h",
    "entradas_lag_24h",
    "entradas_lag_168h",
    "entradas_roll_mean_6h",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "salidas_roll_mean_6h",
    "balance_lag_1h",
    "balance_lag_24h",
    "balance_lag_168h",
    "balance_roll_mean_6h",
]


def build_eda_tables(hourly: pd.DataFrame, features: pd.DataFrame, station_id: int) -> dict:
    """El EDA anual es descriptivo; las correlaciones usan solo enero–septiembre."""
    validate_hourly_grid(hourly)
    if station_id not in hourly.estacion_id.unique():
        raise ValueError(f"No existe la estación {station_id} en el dataset.")
    df = hourly.assign(
        hora=hourly.timestamp.dt.hour,
        dia_semana=hourly.timestamp.dt.dayofweek,
        es_fin_semana=hourly.timestamp.dt.dayofweek.ge(5).astype(int),
        balance_absoluto=hourly.balance.abs(),
    )
    daily = df.groupby(df.timestamp.dt.floor("D"))[["entradas", "salidas"]].sum()
    daily.index.name = "fecha"
    weekdays = (
        daily.groupby(daily.index.dayofweek)
        .agg(
            entradas_total=("entradas", "sum"),
            salidas_total=("salidas", "sum"),
            entradas_media_diaria=("entradas", "mean"),
            salidas_media_diaria=("salidas", "mean"),
            dias_observados=("entradas", "size"),
        )
        .rename_axis("dia_semana")
        .reset_index()
    )
    weekdays["nombre_dia"] = weekdays.dia_semana.map(dict(enumerate(DAYS)))
    monthly = (
        daily.groupby(daily.index.month)
        .agg(
            entradas_total=("entradas", "sum"),
            salidas_total=("salidas", "sum"),
            salidas_media_diaria=("salidas", "mean"),
            dias_observados=("salidas", "size"),
        )
        .rename_axis("mes")
        .reset_index()
    )
    stations = (
        df.groupby("estacion_id")
        .agg(
            total_entradas=("entradas", "sum"),
            total_salidas=("salidas", "sum"),
            balance_neto=("balance", "sum"),
            desbalance_absoluto=("balance_absoluto", "sum"),
        )
        .reset_index()
        .sort_values(["desbalance_absoluto", "estacion_id"], ascending=[False, True])
    )
    hourly_flow = (
        df.groupby(["hora", "es_fin_semana"])[["entradas", "salidas"]].mean().reset_index()
    )
    station_flow = (
        df.loc[(df.estacion_id == station_id) & (df.dia_semana < 5)]
        .groupby("hora")[["entradas", "salidas", "balance"]]
        .mean()
        .reset_index()
    )
    required = {"timestamp", *CORR_COLUMNS}
    if not required.issubset(features.columns):
        raise ValueError(
            f"Faltan columnas en features: {sorted(required - set(features.columns))}"
        )
    train = features.loc[features.timestamp.dt.month <= 9, CORR_COLUMNS]
    if train.empty:
        raise ValueError("No hay datos de enero–septiembre para las correlaciones.")
    corr = train.corr()
    empty = (df.entradas == 0) & (df.salidas == 0)
    equal_with_flow = (df.balance == 0) & ~empty
    station_month = df.groupby(["estacion_id", df.timestamp.dt.month])[
        ["entradas", "salidas"]
    ].sum()
    metrics = {
        "filas_hourly": len(df),
        "columnas_hourly": len(hourly.columns),
        "filas_features": len(features),
        "columnas_features": len(features.columns),
        "estaciones": int(df.estacion_id.nunique()),
        "dias_observados": len(daily),
        "entradas_total": int(df.entradas.sum()),
        "salidas_total": int(df.salidas.sum()),
        "porcentaje_sin_movimientos": float(empty.mean() * 100),
        "porcentaje_balance_cero_con_movimientos": float(equal_with_flow.mean() * 100),
        "estacion_meses_sin_movimientos": int(
            ((station_month.entradas == 0) & (station_month.salidas == 0)).sum()
        ),
        "filas_correlacion_entrenamiento": len(train),
        "estacion_ejemplo": station_id,
    }
    if metrics["entradas_total"] != metrics["salidas_total"]:
        raise ValueError(
            "Los totales de la red deben conservar los viajes de inicio y fin dentro del año."
        )
    return {
        "metricas": metrics,
        "estadisticas": hourly[["entradas", "salidas", "balance"]]
        .describe()
        .T.reset_index(names="variable"),
        "flujo_horario": hourly_flow,
        "dias_semana": weekdays,
        "meses": monthly,
        "estaciones": stations.reset_index(drop=True),
        "estacion_ejemplo": station_flow,
        "correlacion": corr.reset_index(names="variable"),
    }


def configure_visual_style():
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10})


def save_figure(fig, figures_dir: Path, name: str) -> Path:
    fig.tight_layout()
    path = figures_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hourly_flow(tables: dict, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, column, title in zip(axes, ["entradas", "salidas"], ["Entradas", "Salidas"]):
        for weekend, label, color in [
            (0, "Lunes–viernes", "#245684"),
            (1, "Sábado–domingo", "#ca6d22"),
        ]:
            sub = tables["flujo_horario"].query("es_fin_semana == @weekend")
            ax.plot(sub.hora, sub[column], marker=".", label=label, color=color)
        ax.set(title=title, xlabel="Hora del día", xticks=range(0, 24, 2))
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Movimientos medios por estación y hora")
    fig.suptitle("Perfil horario registrado en 2023", y=1.02)
    return save_figure(fig, figures_dir, "01_flujo_horario_pico.png")


def plot_weekday_demand(tables: dict, figures_dir: Path) -> Path:
    data = tables["dias_semana"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    bars = ax.bar(
        data.nombre_dia, data.salidas_media_diaria, color=["#245684"] * 5 + ["#ca6d22"] * 2
    )
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set(
        title="Viajes iniciados por día de semana (2023)",
        ylabel="Promedio diario en toda la red",
        xlabel="Día de la semana",
    )
    ax.margins(y=0.15)
    return save_figure(fig, figures_dir, "02_demanda_dia_semana.png")


def plot_station_imbalance(tables: dict, figures_dir: Path) -> Path:
    top = tables["estaciones"].head(15).sort_values("desbalance_absoluto")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.estacion_id.astype(str), top.desbalance_absoluto, color="#245684")
    ax.set(
        title="15 estaciones con mayor suma de desbalances horarios absolutos",
        xlabel="Suma anual de |entradas − salidas| por hora",
        ylabel="Estación",
    )
    return save_figure(fig, figures_dir, "03_top_estaciones_desbalance.png")


def plot_station_flow(tables: dict, figures_dir: Path) -> Path:
    data = tables["estacion_ejemplo"]
    station = tables["metricas"]["estacion_ejemplo"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for column, color in [("entradas", "#245684"), ("salidas", "#ca6d22")]:
        axes[0].plot(data.hora, data[column], label=column.capitalize(), color=color, marker=".")
    axes[0].set_ylabel("Movimientos medios por hora")
    axes[0].legend()
    axes[0].set_title(f"Estación {station}: perfil de lunes a viernes en 2023")
    axes[1].plot(data.hora, data.balance, color="#3c7e56", marker=".")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Hora del día", ylabel="Balance medio (entradas − salidas)", xticks=range(0, 24)
    )
    return save_figure(fig, figures_dir, f"04_dinamica_estacion_{station}.png")


def plot_correlations(tables: dict, figures_dir: Path) -> Path:
    corr = tables["correlacion"].set_index("variable")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot_kws={"fontsize": 7},
        ax=ax,
    )
    ax.set_title("Flujos, balance y variables históricas: enero–septiembre de 2023")
    return save_figure(fig, figures_dir, "05_matriz_correlacion_lags.png")


def plot_monthly_flow(tables: dict, figures_dir: Path) -> Path:
    data = tables["meses"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(data.mes, data.salidas_media_diaria, color="#245684", marker="o")
    ax.set(
        title="Viajes iniciados por mes (2023)",
        xlabel="Mes",
        ylabel="Promedio diario en toda la red",
        xticks=range(1, 13),
    )
    return save_figure(fig, figures_dir, "06_flujo_mensual.png")


def interpret_tables(tables: dict) -> str:
    m = tables["metricas"]
    hourly = tables["flujo_horario"]
    work = hourly.loc[hourly.es_fin_semana == 0].nlargest(3, "salidas")
    week = tables["dias_semana"]
    weekday_avg = np.average(
        week.loc[week.dia_semana < 5, "salidas_media_diaria"],
        weights=week.loc[week.dia_semana < 5, "dias_observados"],
    )
    weekend_avg = np.average(
        week.loc[week.dia_semana >= 5, "salidas_media_diaria"],
        weights=week.loc[week.dia_semana >= 5, "dias_observados"],
    )
    months = tables["meses"]
    highest = months.loc[months.salidas_media_diaria.idxmax()]
    lowest = months.loc[months.salidas_media_diaria.idxmin()]
    station = tables["estaciones"].iloc[0]
    return (
        f"Se analizaron {m['salidas_total']:,} viajes conservados, {m['estaciones']} estaciones y "
        f"{m['dias_observados']} días. Los totales de entradas y salidas son iguales por construcción "
        "del conjunto, que exige que ambos extremos del viaje estén dentro de 2023.\n\n"
        f"El {m['porcentaje_sin_movimientos']:.2f}% de las combinaciones estación-hora no registra "
        f"movimientos. Otro {m['porcentaje_balance_cero_con_movimientos']:.2f}% tiene balance cero "
        "con entradas y salidas positivas. Un balance cero no implica ausencia de viajes.\n\n"
        f"Las tres horas con más salidas medias en lunes–viernes son "
        f"{', '.join(str(int(v)) + ':00' for v in work.hora)}. El promedio diario de salidas es "
        f"{weekday_avg:,.0f} de lunes a viernes y {weekend_avg:,.0f} en fin de semana "
        f"({(1 - weekend_avg / weekday_avg) * 100:.1f}% menos). Se comparan promedios por día "
        "para evitar el efecto de tener distinto número de lunes, martes, etc.\n\n"
        f"El mes {int(highest.mes)} tiene el mayor promedio diario de salidas "
        f"({highest.salidas_media_diaria:,.0f}); el mes {int(lowest.mes)}, el menor "
        f"({lowest.salidas_media_diaria:,.0f}). Son patrones descriptivos; no prueban causas "
        "como clima, cierres o cambios en la oferta.\n\n"
        f"La estación {int(station.estacion_id)} presenta la mayor suma de desbalances horarios "
        f"absolutos ({int(station.desbalance_absoluto):,}). Este indicador también depende del "
        "volumen de uso y no demuestra urgencia de rebalanceo: no tenemos inventario ni capacidad. "
        f"Hay {m['estacion_meses_sin_movimientos']} combinaciones estación-mes sin movimientos; "
        "no se conocen sus fechas reales de apertura, cierre o fallos.\n\n"
        "Las correlaciones se calcularon solo con enero–septiembre e incluyen las medias móviles "
        "reales *_roll_mean_6h. No prueban causalidad ni desempeño predictivo. La relación del "
        "balance con entradas y salidas es, en parte, consecuencia de su definición.\n\n"
        "Se mantienen los conteos en sus unidades y la codificación cíclica de hora y día de semana. "
        "No se aplican escalado ni PCA: el modelo de árboles existente no requiere escalado y se "
        "prioriza interpretar las variables. La estación numérica y la bandera de hora pico son "
        "decisiones provisionales. Las variables históricas requieren que las horas anteriores "
        "ya se hayan observado; este EDA no acredita predicciones de varios pasos ni capacidad "
        "para anticipar bicicletas disponibles. No se reentrenaron modelos.\n"
    )


def run_eda(
    input_path: Path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet",
    features_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
    figures_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORTS_DIR / "eda",
    tracking_uri: str = MLFLOW_TRACKING_URI,
    station_id: int = 51,
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with data_run("plots", tracking_uri) as run_id:
        mlflow.log_params(
            {
                "estacion_ejemplo": station_id,
                "correlacion_meses": "1-9",
                "comparacion_temporal": "promedio_diario",
                "periodo_descriptivo": "2023",
            }
        )
        hourly = pd.read_parquet(input_path)
        features = pd.read_parquet(features_path, columns=["timestamp", *CORR_COLUMNS])
        tables = build_eda_tables(hourly, features, station_id)
        # La lectura parcial reduce memoria; las dimensiones completas proceden del esquema Parquet.
        tables["metricas"]["columnas_features"] = pq.read_metadata(features_path).num_columns
        log_dataframe(hourly, input_path, "eda_horario")
        log_dataframe(features, features_path, "eda_correlacion")
        configure_visual_style()
        figures = [
            plot(tables, figures_dir)
            for plot in [
                plot_hourly_flow,
                plot_weekday_demand,
                plot_station_imbalance,
                plot_station_flow,
                plot_correlations,
                plot_monthly_flow,
            ]
        ]
        serialized = {
            name: json.loads(table.to_json(orient="records"))
            for name, table in tables.items()
            if isinstance(table, pd.DataFrame)
        }
        tables_path = report_dir / f"tablas_{run_id}.json"
        tables_path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        findings = interpret_tables(tables)
        findings_path = report_dir / f"hallazgos_{run_id}.txt"
        findings_path.write_text(findings, encoding="utf-8")
        report = {
            "metricas": tables["metricas"],
            "entradas": [file_metadata(input_path), file_metadata(features_path)],
            "figuras": [file_metadata(path) for path in figures],
            "tablas": str(tables_path.resolve()),
            "hallazgos": str(findings_path.resolve()),
            "reglas": [
                "EDA anual descriptivo; no selecciona automáticamente características.",
                "Correlaciones de enero a septiembre, sin usar octubre–diciembre.",
                "Balance cero se distingue de entradas y salidas simultáneamente cero.",
                "Desbalance histórico no equivale a inventario ni a urgencia operativa.",
            ],
        }
        for path in figures:
            mlflow.log_artifact(str(path), artifact_path="figuras")
        for path in [tables_path, findings_path]:
            mlflow.log_artifact(str(path), artifact_path="eda")
        save_report(report, report_dir, "resumen", run_id)
    logger.success(f"EDA terminado: {run_id}")
    return report


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / f"hourly_balance_{TARGET_YEAR}.parquet",
    features_path: Path = PROCESSED_DATA_DIR / f"features_{TARGET_YEAR}.parquet",
    figures_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORTS_DIR / "eda",
    tracking_uri: str = MLFLOW_TRACKING_URI,
    station_id: int = 51,
):
    run_eda(input_path, features_path, figures_dir, report_dir, tracking_uri, station_id)


if __name__ == "__main__":
    app()
