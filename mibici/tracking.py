"""Registro compartido de las etapas de datos, sin subir los datasets a MLflow."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from loguru import logger
import mlflow
import pandas as pd

from mibici.config import MLFLOW_EXPERIMENT_NAME, PROJ_ROOT


def file_metadata(path: Path) -> dict:
    """Identifica el contenido completo del archivo, independientemente de su ubicación."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"archivo": path.name, "ruta_local": str(path.resolve()), "sha256": digest.hexdigest()}


@contextmanager
def data_run(stage: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=stage) as run:
        mlflow.set_tag("etapa", stage)
        mlflow.set_tag("responsable_del_flujo", "Leo")
        # HEAD identifica código confirmado; los fuentes adjuntos incluyen cambios locales.
        for command, tag in [
            (["git", "rev-parse", "HEAD"], "codigo_commit"),
            (["git", "status", "--porcelain"], "codigo_cambios_locales"),
        ]:
            result = subprocess.run(command, cwd=PROJ_ROOT, capture_output=True, text=True)
            value = result.stdout.strip() if result.returncode == 0 else "no_disponible"
            mlflow.set_tag(tag, str(bool(value)) if tag.endswith("locales") else value)
        for source in [
            PROJ_ROOT / "mibici" / f"{stage}.py",
            PROJ_ROOT / "mibici" / "config.py",
            PROJ_ROOT / "mibici" / "tracking.py",
            PROJ_ROOT / "requirements.txt",
        ]:
            mlflow.log_artifact(str(source), artifact_path="codigo")
        mlflow.log_dict(
            {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "mlflow": mlflow.__version__,
            },
            "entorno.json",
        )
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        mlflow.log_text(installed.stdout, "entorno_requirements.txt")
        logger.info(f"MLflow run_id ({stage}): {run.info.run_id}")
        yield run.info.run_id


def log_dataframe(df: pd.DataFrame, path: Path, context: str):
    """Registra esquema y huella MLflow; el SHA-256 del archivo va en el reporte."""
    dataset = mlflow.data.from_pandas(df, source=str(path.resolve()), name=path.stem)
    mlflow.log_input(dataset, context=context)


def save_report(report: dict, report_dir: Path, stage: str, run_id: str) -> Path:
    report["run_id"] = run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{stage}_{run_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(path), artifact_path="calidad")
    mlflow.log_metrics(report["metricas"])
    return path
