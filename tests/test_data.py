"""Pruebas pequeñas: reglas de limpieza, conservación de conteos y trazabilidad."""

import json

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici.config import MLFLOW_EXPERIMENT_NAME
from mibici.dataset import app as dataset_app
from mibici.dataset import (
    build_hourly_grid,
    clean_trips,
    monthly_files,
    parse_trip_timestamps,
)
from mibici.features import (
    add_calendar_and_cyclical_features,
    add_historical_lags_and_rolling,
)
from mibici.features import app as features_app
from mibici.features import validate_hourly_grid
from mibici.plots import app as plots_app
from mibici.tracking import file_metadata


def trip(trip_id=1, **changes):
    row = {
        "Viaje_Id": trip_id,
        "Origen_Id": 1,
        "Destino_Id": 2,
        "Inicio_del_viaje": "2023-01-01 00:00:00",
        "Fin_del_viaje": "2023-01-01 00:30:00",
        "Genero": None,
    }
    return row | changes


def test_cleaning_accounts_for_every_discard_without_requiring_demographics():
    rows = [
        trip(),
        trip(),
        trip(2, Destino_Id=1, Fin_del_viaje="2023-01-01 00:00:30"),
        trip(3, Fin_del_viaje="2023-01-03 00:00:00"),
        trip(4, Origen_Id=1.5),
        trip(5, Origen_Id=None, Inicio_del_viaje="incorrecto"),
        trip(6, Inicio_del_viaje="incorrecto"),
        trip(7, Fin_del_viaje="2022-12-31 23:59:00"),
        trip(8, Inicio_del_viaje="2022-12-31 23:59:00"),
        trip(9, Fin_del_viaje=None),
    ]
    cleaned, report = clean_trips(pd.DataFrame(rows), 2023)
    metrics = report["metricas"]
    assert list(cleaned["Viaje_Id"]) == [1, 2, 3]
    assert metrics["descartados_identificadores_invalidos"] == 2
    assert metrics["descartados_fechas_invalidas"] == 2
    assert metrics["descartados_duracion_no_positiva"] == 1
    assert metrics["descartados_fuera_del_anio"] == 1
    assert metrics["descartados_duplicados"] == 1
    assert metrics["descartados_total"] == 7
    assert metrics["filas_entrada"] == metrics["filas_limpias"] + metrics["descartados_total"]
    assert metrics["viajes_menores_60s_misma_estacion"] == 1
    assert metrics["viajes_mayores_24h"] == 1
    assert cleaned["Genero"].isna().all()


def test_conflicting_trip_ids_stop_processing():
    rows = [trip(), trip(Destino_Id=3)]
    with pytest.raises(ValueError, match="movimientos distintos"):
        clean_trips(pd.DataFrame(rows), 2023)


def test_dates_use_explicit_iso_or_day_month_year_order():
    values = pd.Series(["01/11/2023 0:01", "15/11/2023 12:30:45", "2023-01-11 00:01:00"])
    parsed = parse_trip_timestamps(values)
    assert parsed.iloc[0] == pd.Timestamp("2023-11-01 00:01")
    assert parsed.iloc[1] == pd.Timestamp("2023-11-15 12:30:45")
    assert parsed.iloc[2] == pd.Timestamp("2023-01-11 00:01")
    cleaned, report = clean_trips(
        pd.DataFrame(
            [
                trip(Inicio_del_viaje="31/10/2023 23:59", Fin_del_viaje="01/11/2023 0:01"),
            ]
        ),
        2023,
    )
    assert len(cleaned) == 1
    assert report["duracion_segundos"]["min"] == 120


def test_missing_month_is_not_imputed_as_zero(tmp_path):
    with pytest.raises(FileNotFoundError, match="2023_04.csv"):
        monthly_files(tmp_path, 2023)


@pytest.mark.parametrize("year,hours", [(2023, 8760), (2024, 8784)])
def test_grid_conserves_counts_and_supports_leap_years(year, hours):
    trips, _ = clean_trips(
        pd.DataFrame(
            [
                trip(
                    Inicio_del_viaje=f"{year}-01-01 00:00:00",
                    Fin_del_viaje=f"{year}-01-01 00:30:00",
                ),
            ]
        ),
        year,
    )
    grid = build_hourly_grid(trips, year)
    assert len(grid) == 2 * hours
    assert grid["entradas"].sum() == grid["salidas"].sum() == 1
    assert grid["balance"].sum() == 0
    assert grid.loc[0, "balance"] == -1
    assert grid.loc[hours, "balance"] == 1
    assert not grid.duplicated(["estacion_id", "timestamp"]).any()


def sample_grid():
    frames = []
    for station in [1, 2]:
        values = np.arange(180) + station * 1000
        frames.append(
            pd.DataFrame(
                {
                    "estacion_id": station,
                    "timestamp": pd.date_range("2023-01-01", periods=180, freq="h"),
                    "entradas": values + 1,
                    "salidas": values,
                    "balance": 1,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_history_uses_only_past_values_and_respects_station():
    grid = sample_grid()
    validate_hourly_grid(grid)
    features = add_historical_lags_and_rolling(add_calendar_and_cyclical_features(grid.copy()))
    assert len(features) == 2 * (180 - 168)
    assert len(features.columns) == 33
    for station in [1, 2]:
        row = features.loc[features["estacion_id"] == station].iloc[0]
        assert row["salidas_lag_168h"] == station * 1000
        assert row["salidas_roll_mean_3h"] == station * 1000 + 166
        assert row["salidas_roll_mean_6h"] == station * 1000 + 164.5
    # Alterar el futuro no debe cambiar las características de horas anteriores.
    future = grid["timestamp"] >= pd.Timestamp("2023-01-08 02:00")
    changed = grid.copy()
    changed.loc[future, ["entradas", "salidas"]] += 50000
    recomputed = add_historical_lags_and_rolling(add_calendar_and_cyclical_features(changed))
    before = features["timestamp"] < pd.Timestamp("2023-01-08 02:00")
    pd.testing.assert_frame_equal(features.loc[before], recomputed.loc[before])


@pytest.mark.parametrize("problem", ["hole", "duplicate", "balance"])
def test_invalid_grid_is_rejected(problem):
    grid = sample_grid()
    if problem == "hole":
        grid = grid.drop(index=20)
    elif problem == "duplicate":
        grid = pd.concat([grid, grid.iloc[[0]]])
    else:
        grid.loc[0, "balance"] = 999
    with pytest.raises(ValueError):
        validate_hourly_grid(grid)


def test_cli_pipeline_registers_two_finished_mlflow_runs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for month in range(1, 13):
        row = trip(
            month,
            Inicio_del_viaje=f"2023-{month:02d}-01 00:00:00",
            Fin_del_viaje=f"2023-{month:02d}-01 00:30:00",
        )
        if month == 11:
            row.update(Inicio_del_viaje="01/11/2023 0:00", Fin_del_viaje="01/11/2023 0:30")
        pd.DataFrame([row]).to_csv(
            raw / f"datos_abiertos_2023_{month:02d}.csv", index=False, encoding="latin1"
        )
    hourly = tmp_path / "hourly.parquet"
    features = tmp_path / "features.parquet"
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        MLFLOW_EXPERIMENT_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    common = ["--tracking-uri", uri, "--report-dir", str(reports)]
    runner = CliRunner()
    try:
        result = runner.invoke(
            dataset_app,
            [
                "--raw-dir",
                str(raw),
                "--output-path",
                str(hourly),
                *common,
            ],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)
        grid = pd.read_parquet(hourly)
        november = grid[(grid.estacion_id == 1) & (grid.timestamp == pd.Timestamp("2023-11-01"))]
        assert november.salidas.iloc[0] == 1
        result = runner.invoke(
            features_app,
            [
                "--input-path",
                str(hourly),
                "--output-path",
                str(features),
                *common,
            ],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)
        result = runner.invoke(
            plots_app,
            [
                "--input-path",
                str(hourly),
                "--features-path",
                str(features),
                "--figures-dir",
                str(figures),
                "--station-id",
                "1",
                *common,
            ],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)
        assert len(list(figures.glob("*.png"))) == 6
        runs = client.search_runs([experiment_id])
        assert len(runs) == 3
        assert all(run.info.status == "FINISHED" for run in runs)
        stages = {run.data.tags["etapa"]: run for run in runs}
        assert stages["dataset"].data.metrics["filas_limpias"] == 12
        assert stages["features"].data.metrics["filas_sin_historial"] == 2 * 168
        for stage, run in stages.items():
            assert run.inputs.dataset_inputs
            artifacts = client.list_artifacts(run.info.run_id, "calidad")
            assert len(artifacts) == 1
            prefix = "resumen" if stage == "plots" else stage
            report = json.loads((reports / f"{prefix}_{run.info.run_id}.json").read_text())
            if stage == "plots":
                assert report["metricas"]["salidas_total"] == 12
                assert len(client.list_artifacts(run.info.run_id, "figuras")) == 6
                continue
            output = hourly if stage == "dataset" else features
            assert report["salida"]["sha256"] == file_metadata(output)["sha256"]
    finally:
        mlflow.set_tracking_uri(previous_uri)
