from pathlib import Path

# Directorio raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Rutas de datos estructuradas
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de modelos y reportes
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Asegurar directorios de salida
for directory in [INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Parámetros del proyecto
TARGET_YEAR = 2023
MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT_NAME = "mibici_balance_estaciones_2023"