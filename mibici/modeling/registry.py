"""
Gestion de versiones, etiquetas y carga controlada desde MLflow Model Registry
para el predictor de balance horario MiBici.
"""

from typing import Any, Dict, Optional

from loguru import logger
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from mibici.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


def register_model(
    model_uri: str,
    model_name: str = REGISTERED_MODEL_NAME,
    tags: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Registra un artefacto de run en MLflow Model Registry y asigna tags de trazabilidad."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    version_str = str(model_version.version)

    if description:
        client.update_model_version(
            name=model_name,
            version=version_str,
            description=description,
        )

    default_tags = {
        "stage": "candidate",
        "dataset": "v2",
        "validation_passed": "true",
    }
    if tags:
        default_tags.update({k: str(v) for k, v in tags.items()})

    for key, value in default_tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version_str,
            key=key,
            value=value,
        )

    logger.success(f"Modelo '{model_name}' registrado en version v{version_str} desde {model_uri}")
    return model_version


def get_model_version(
    model_name: str = REGISTERED_MODEL_NAME,
    version: str | int = 1,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> Dict[str, Any]:
    """Consulta estado, run_id y etiquetas de una version especifica del Registry."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version_str = str(version)

    mv = client.get_model_version(name=model_name, version=version_str)
    if mv.status != "READY":
        raise RuntimeError(
            f"La version v{version_str} de '{model_name}' no esta lista (status={mv.status})."
        )

    return {
        "model_name": mv.name,
        "version": mv.version,
        "status": mv.status,
        "run_id": mv.run_id,
        "source": mv.source,
        "tags": dict(mv.tags),
        "model_uri": f"models:/{mv.name}/{mv.version}",
    }


def load_registered_model(
    model_name: str = REGISTERED_MODEL_NAME,
    version: str | int = 1,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Verifica el estado de una version explicita y carga el pipeline desde el Registry."""
    info = get_model_version(
        model_name=model_name,
        version=version,
        tracking_uri=tracking_uri,
    )
    logger.info(f"Cargando modelo verificado desde {info['model_uri']} (run={info['run_id']})")
    return mlflow.sklearn.load_model(info["model_uri"])
