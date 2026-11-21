"""Verificacion de integridad y gobernanza de datasets versionados contra manifiestos."""

import json
from pathlib import Path

from mibici.config import PROJ_ROOT
from mibici.tracking import file_metadata

MANIFEST_V2_PATH = PROJ_ROOT / "manifests" / "mibici_processed_v2.json"


def verify_dataset_manifest(
    dataset_path: Path,
    manifest_id: str = "features_2023_v2",
    manifest_path: Path = MANIFEST_V2_PATH,
) -> dict:
    """Verifica que el archivo coincida exactamente con el SHA-256 registrado en el manifiesto."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifiesto no encontrado: {manifest_path}")

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in entries if item["id"] == manifest_id), None)
    if entry is None:
        raise KeyError(f"ID '{manifest_id}' no existe en {manifest_path.name}")

    meta = file_metadata(dataset_path)
    expected_sha = entry["sha256"]
    if meta["sha256"] != expected_sha:
        raise ValueError(
            f"Integridad comprometida para {dataset_path.name}. "
            f"SHA-256 esperado ({manifest_id}): {expected_sha} | Actual: {meta['sha256']}. "
            "Crea una nueva version de manifiesto antes de entrenar."
        )
    return {**entry, "ruta_local": meta["ruta_local"]}
