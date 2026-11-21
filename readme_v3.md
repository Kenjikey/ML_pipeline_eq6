# MiBici Guadalajara — Pipeline MLOps de Predicción Horaria de Balance y Rebalanceo (v3)

Sistema end-to-end de **Machine Learning Operations (MLOps)** desarrollado sobre la estructura **Cookiecutter Data Science (`ccds`)** para predecir la demanda horaria de entradas, salidas y balance neto ($\text{Balance} = \text{Entradas} - \text{Salidas}$) por estación del sistema de bicicletas públicas **MiBici** (Área Metropolitana de Guadalajara), emitiendo alertas operativas de rebalanceo en tiempo real.

---

## 1. Arquitectura del Repositorio

```text
ML_pipeline_eq6/
├── .github/workflows/
│   └── tests.yml                     # CI en GitHub Actions (flake8, isort, black y pytest)
├── configs/
│   ├── data.yaml                     # Rutas de datos crudos, intermedios, procesados y manifiestos
│   ├── model.yaml                    # Split temporal, baseline heurístico y candidatos HGB
│   ├── serving.yaml                  # Configuración del servicio FastAPI, URIs y umbrales
│   └── drift.yaml                    # Parámetros de monitoreo PSI/KS y simulación de drift
├── data/
│   ├── raw/                          # CSV mensuales originales (datos_abiertos_2023_01..12.csv)
│   ├── interim/                      # Malla horaria continua (hourly_balance_2023.parquet)
│   └── processed/                    # Dataset v2 con 29 features (features_2023_v2.parquet)
├── manifests/
│   ├── mibici_original.json          # Hashes SHA-256 de los 12 CSV crudos
│   └── mibici_processed_v2.json      # Hash SHA-256 oficial del dataset v2 procesado
├── mibici/
│   ├── config.py                     # Rutas globales, constantes MLflow y lector de YAML
│   ├── integrity.py                  # Verificación criptográfica SHA-256 contra manifiestos
│   ├── tracking.py                   # Helpers de trazabilidad y registro de metadatos
│   ├── dataset.py                    # Limpieza secuencial y construcción de malla horaria 1h
│   ├── features.py                   # Ingeniería de 29 variables (calendario, cíclicas, lags, rolling)
│   ├── pipeline.py                   # ColumnTransformer + MultiOutputRegressor(HistGradientBoosting)
│   ├── modeling/
│   │   ├── evaluate.py               # Métricas MAE/RMSE/R²/MAPE, balance neto y BaselineHeuristicModel
│   │   ├── train.py                  # Split fuera de tiempo, selección en Val, compuerta y registro
│   │   ├── predict.py                # Inferencia batch y reglas de acción operativa (±2.0 bicis)
│   │   └── registry.py               # Registro, etiquetado y carga desde MLflow Model Registry
│   ├── monitoring/
│   │   └── drift.py                  # Detección de Data Drift (PSI/KS) y Model Degradation
│   └── serving/
│       ├── schemas.py                # Contratos estrictos Pydantic v2 (extra="forbid")
│       └── app.py                    # API REST FastAPI (/health, /model-info, /schema, /predict)
├── models/
│   ├── mibici_balance_pipeline.joblib # Artefacto serializado del pipeline consolidado
│   └── model_schema.json             # Contrato de esquema, dtypes, commit y métricas de evaluación
├── reports/
│   ├── quality/                      # Reportes JSON de auditoría de limpieza y features
│   └── monitoring/                   # Reportes JSON de monitoreo de drift y alertas
├── tests/
│   ├── test_data.py                  # Pruebas de limpieza, fechas ambiguas y malla continua
│   ├── test_modeling.py              # Pruebas de pipeline, baseline, split temporal y acciones
│   ├── test_serving.py               # Pruebas de contrato HTTP (200, 422, 503) en FastAPI
│   ├── test_drift.py                 # Pruebas de cálculo PSI y alertas por drift sintético
│   └── test_integration_pipeline.py  # Prueba end-to-end (crudos -> malla -> features -> modelo)
├── Dockerfile                        # Imagen ligera python:3.11-slim con libgomp1 y usuario no-root
├── .dockerignore                     # Exclusión de datos, mlruns y artefactos temporales
├── Makefile                          # Atajos operativos (lint, format, data, features, train, test)
├── pyproject.toml                    # Metadatos del paquete y configuración de ruff/black/isort
└── requirements.txt                  # Dependencias fijadas del entorno
```

---

## 2. Requisitos e Instalación

1. **Crear y activar entorno virtual (Python 3.11):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias y el paquete local `mibici` en modo editable:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e . --no-deps
   ```

---

## 3. Flujo Operativo Paso a Paso

### Paso 1: Levantar servidor local de MLflow Tracking & Model Registry
En una terminal dedicada, inicia el servidor de MLflow en el puerto `5001` (configurado por defecto en `mibici/config.py`):

```powershell
mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
*Interfaz web disponible en:* `http://127.0.0.1:5001`

---

### Paso 2: Procesamiento de Datos e Ingeniería de Características (`v2`)
1. **Limpieza y construcción de la malla horaria continua** (12 meses de 2023, resolución `1h`, conservación exacta de movimientos):
   ```powershell
   python -m mibici.dataset
   ```
   *Salida:* `data/interim/hourly_balance_2023.parquet` y reporte en `reports/quality/`.

2. **Generación de las 29 variables explicativas** (sin *data leakage*, eliminando las primeras 168 horas por ventana de inicialización):
   ```powershell
   python -m mibici.features --output-path data/processed/features_2023_v2.parquet
   ```
   *Variables incluidas:*
   * **Categórica:** `estacion_id` (codificada con `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`).
   * **Calendario y cíclicas:** `mes`, `dia_mes`, `dia_semana`, `hora`, `es_fin_de_semana`, `es_hora_pico`, `hora_sin`, `hora_cos`, `dia_semana_sin`, `dia_semana_cos`.
   * **Lags históricos ($t-1\text{h}, t-2\text{h}, t-24\text{h}, t-168\text{h}$):** para `entradas`, `salidas` y `balance`.
   * **Medias móviles desplazadas 1h ($3\text{h}, 6\text{h}$):** para `entradas`, `salidas` y `balance`.

---

### Paso 3: Entrenamiento, Validación Fuera de Tiempo y Model Registry
El módulo `mibici/modeling/train.py` ejecuta el ciclo completo de modelado gobernado:

1. **Verificación de integridad SHA-256:** Valida `features_2023_v2.parquet` contra `manifests/mibici_processed_v2.json` antes de leer datos.
2. **Partición cronológica estricta (Out-of-Time):**
   * **Train:** Enero – Septiembre (`mes <= 9`)
   * **Validation:** Octubre (`mes == 10`) — selección de hiperparámetros (`select_by_validation`).
   * **Test:** Noviembre – Diciembre (`mes >= 11`) — evaluación única final.
3. **Compuerta de calidad (*Quality Gate*):** Verifica que las predicciones sean finitas y que el modelo supere al `BaselineHeuristicModel` (mediana histórica por estación, día de semana y hora) en validación (`val_balance_mae < val_mae_heur`) antes de registrar.
4. **Exportación y Registro:** Guarda `models/mibici_balance_pipeline.joblib`, `models/model_schema.json` y registra una nueva versión etiquetada (`stage=candidate`, `dataset=v2`, `git_commit`, `dataset_sha256`) en **MLflow Model Registry** bajo el nombre `mibici_balance_predictor`.

```powershell
python -m mibici.modeling.train
```

---

### Paso 4: Monitoreo de Data Drift y Degradación
Evalúa desplazamientos de distribución mediante **PSI (*Population Stability Index*)** y prueba de **Kolmogorov-Smirnov (`ks_2samp`)** sobre las variables históricas críticas, además de detectar degradación en `balance_mae`:

```powershell
python -m mibici.monitoring.drift
```
*Salida:* Reporte JSON estructurado con estado (`OK` / `ALERT`) y listado de alertas en `reports/monitoring/drift_report_<timestamp>.json`.

---

## 4. Servicio de Inferencia (FastAPI) y Docker

### Ejecución local con Uvicorn
```powershell
uvicorn mibici.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints disponibles
| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Estado de salud del servicio, carga del modelo en memoria y versión activa. |
| `GET` | `/model-info` | Metadatos del modelo registrado, `dataset_id`, hash y listado de 29 features. |
| `GET` | `/schema` | Alias del contrato de esquema y metadatos del artefacto cargado. |
| `POST` | `/predict` | Predicción individual para un objeto `StationHourFeatures` (`extra="forbid"`). |
| `POST` | `/predict/batch` | Predicción por lote (`{"records": [...]}` con máximo 1,000 registros). |

#### Reglas de Acción Operativa
* $\text{Balance predicho} \ge +2.0 \rightarrow$ `ALERTA SUPERAVIT (Retirar Bicis)`
* $\text{Balance predicho} \le -2.0 \rightarrow$ `ALERTA DEFICIT (Abastecer Bicis)`
* $-2.0 < \text{Balance predicho} < +2.0 \rightarrow$ `ESTABLE`

### Construcción y despliegue con Docker
```powershell
# 1. Construir imagen de producción
docker build -t mibici-api:2.0.0 .

# 2. Iniciar contenedor en segundo plano
docker run -d --name mibici-serving -p 8000:8000 mibici-api:2.0.0

# 3. Verificar estado de salud
curl.exe http://localhost:8000/health
```

---

## 5. Calidad de Código y Pruebas Automáticas (CI)

Antes de confirmar cambios en Git, valida formato, linting y la suite completa de pruebas unitarias e integración:

```powershell
# Formatear código
python -m isort mibici tests
python -m black mibici tests

# Verificar reglas de estilo (máx. 99 caracteres por línea)
python -m flake8 --max-line-length=99 mibici tests

# Ejecutar suite completa de pruebas
python -m pytest -v
```