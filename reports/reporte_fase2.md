# MLOps Pipeline: Rebalanceo Predictivo de Flota (MiBici Guadalajara 2023)
## Reporte Técnico de Avance · Fase 2 (Gestión del Proyecto de Machine Learning)

**Equipo 6:**  
- **Leonel Matus Climaco** 
- **Kenji Ramírez** 
- **Iván V.** 

**Repositorio GitHub:** `https://github.com/Kenjikey/ML_pipeline_eq6.git`  
**Rama Activa:** `feature/mlops-pipeline-2023`  
**Diseñado Para:** Personal de Operación y Planeación de MiBici  
**Fecha de Entrega:** Noviembre 2026  
**Iteración:** Fase 2 (Refactorización Modular, Scikit-Learn Pipelines y MLflow Model Registry)

========================================================================================

## 1. Resumen Ejecutivo de la Fase 2

En esta segunda fase, el Equipo 6 transformó el prototipo exploratorio de la Fase 1 en un **sistema de Machine Learning modular, estandarizado, versionado y reproducible para producción**.

### Logros Principales de la Fase 2:
1. **Adopción de la Plantilla Cookiecutter Data Science:** Estructuración estandarizada del proyecto separando datos (`raw`, `interim`, `processed`), código modular (`src/mibici_mlops`), modelos, reportes y pruebas automatizadas.
2. **Refactorización Completa:** Extracción de toda la lógica de notebooks hacia módulos especializados de Python (`dataset.py`, `features.py`, `pipeline.py`, `train.py`, `predict.py`, `evaluate.py`) con configuración centralizada en `config/params.yaml`.
3. **Pipeline de Scikit-Learn de Extremo a Extremo:** Implementación de `ColumnTransformer`, transformadores personalizados de variables cíclicas (`sin/cos`) y escaladores robustos que garantizan que el preprocesamiento viaje acoplado con el estimador.
4. **Optimización y Tuning de Modelos:** Entrenamiento y ajuste de hiperparámetros para **LightGBM** y **Random Forest**, logrando un **MAE de 0.4721** en el Balance Neto, lo que representa una **reducción de error del +32.22%** frente al Baseline Heurístico.
5. **Gobierno y MLOps con MLflow Tracking & Model Registry:** Registro sistemático de corridas, hiperparámetros, métricas y artefactos con versionado formal de modelos en el registro de MLflow (`mibici_inflow_model` y `mibici_outflow_model`).
6. **Reproducibilidad Garantizada:** Automatización completa mediante `Makefile` (`make data`, `make features`, `make train`, `make predict`, `make test`).

========================================================================================

## 2. Estructuración del Proyecto con Cookiecutter Data Science

Se adoptó la estructura estándar de Cookiecutter Data Science para asegurar la mantenibilidad y escalabilidad del código:

```text
├── Makefile                <- Comandos de automatización del ciclo de vida
├── README.md               <- Documentación principal de arquitectura y uso
├── requirements.txt        <- Dependencias congeladas
├── config/
│   └── params.yaml         <- Hiperparámetros, semillas, rutas y ventanas temporales
├── data/
│   ├── raw/                <- Archivos mensuales crudos de viajes MiBici 2023
│   ├── interim/            <- Cuadrícula horaria continua limpia por estación
│   └── processed/          <- Dataset con features cíclicas, lags y rolling stats
├── models/                 <- Modelos entrenados serializados (.joblib)
├── notebooks/
│   └── 1.0-eda.ipynb       <- Notebook interactivo para visualización y EDA
├── reports/
│   ├── figures/            <- Gráficas de métricas y feature importance
│   └── reporte_fase2.md    <- Reporte técnico de la Fase 2
├── src/
│   └── mibici_mlops/       <- Paquete de Python refactorizado
│       ├── __init__.py
│       ├── utils.py        <- Logger, semillas y carga segura de configuración
│       ├── dataset.py      <- Ingesta, filtrado de anomalías y cuadrícula continua
│       ├── features.py     <- Encoders cíclicos, lags temporales y medias móviles
│       ├── pipeline.py     <- Pipeline formal de Scikit-Learn (ColumnTransformer)
│       └── modeling/
│           ├── __init__.py
│           ├── evaluate.py <- Métricas de regresión y Baseline Heurístico
│           ├── train.py    <- Entrenamiento, tracking en MLflow y Model Registry
│           └── predict.py  <- Inferencia desacoplada y clasificación de alertas
└── tests/
    ├── __init__.py
    ├── test_dataset.py     <- Pruebas unitarias de limpieza y cuadrícula
    └── test_features.py    <- Pruebas unitarias de features y lags
```

========================================================================================

## 3. Refactorización del Código

Se eliminó por completo el código monolítico de los notebooks, trasladando la funcionalidad a módulos especializados con principios SOLID:

### 3.1 Configuración Centralizada (`config/params.yaml`)
Se eliminaron los valores fijos (*hardcoded*) en el código. Las semillas aleatorias (`seed: 42`), rutas de almacenamiento, hiperparámetros de LightGBM/Random Forest y cortes temporales se gestionan exclusivamente desde `config/params.yaml`.

### 3.2 Módulos Desarrollados
* **`src/mibici_mlops/dataset.py`:**
  - Limpia viajes con duraciones menores a 60 segundos (desanclajes accidentales) y mayores a 2 horas.
  - Construye la cuadrícula cartesiana completa de 359 estaciones $\times$ 8,760 horas del año 2023, asegurando que las horas sin actividad queden explícitamente registradas como 0.
* **`src/mibici_mlops/features.py`:**
  - Codifica variables cíclicas ($\sin/\cos$) para hora, día de la semana y mes.
  - Genera retardos temporales (*lags* de 1h, 2h, 3h, 24h, 168h) agrupados por estación para evitar fuga de información (*data leakage*).
  - Calcula medias y desviaciones móviles (*rolling windows* de 3h, 6h y 24h) desplazadas (`shift(1)`) para evitar mirar hacia el futuro.
* **`src/mibici_mlops/modeling/predict.py`:**
  - Carga los modelos entrenados y clasifica el estado operativo de cada estación en:
    * `DEFICIT_CRITICO` (Balance $\le -3$): Requiere camión de abastecimiento.
    * `SUPERAVIT_CRITICO` (Balance $\ge +3$): Requiere camión de retiro.
    * `EQUILIBRADO` ($-2 \le Balance \le +2$): Flujo balanceado.

========================================================================================

## 4. Pipeline de Scikit-Learn y Mejores Prácticas

En el módulo `src/mibici_mlops/pipeline.py`, se integró un `ColumnTransformer` formal de Scikit-Learn:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", RobustScaler(), lag_cols + rolling_cols),
        ("cyclical_passthrough", "passthrough", cyclical_cols),
        ("cat_passthrough", "passthrough", categorical_cols)
    ],
    remainder="drop"
)
```

### Ventajas de esta Arquitectura:
1. **Prevención de Data Leakage:** El escalado estadístico (`RobustScaler`) se ajusta (*fit*) únicamente con el conjunto de entrenamiento (Enero a Septiembre) y solo se aplica (*transform*) en Validación y Test.
2. **Portabilidad Total:** El pipeline serializa tanto las transformaciones matemáticas como el modelo final en un único artefacto `.joblib`, simplificando la inferencia en producción.

========================================================================================

## 5. Construcción, Ajuste y Evaluación de Modelos

### 5.1 Estrategia de División Temporal
Para respetar la naturaleza temporal del problema, se utilizó una división cronológica estricta sin mezcla aleatoria (*no shuffling*):
* **Entrenamiento:** 01 de Enero de 2023 al 30 de Septiembre de 2023 (~75% de los datos).
* **Validación:** 01 de Octubre de 2023 al 15 de Noviembre de 2023 (~12.5% de los datos).
* **Prueba (Test):** 16 de Noviembre de 2023 al 31 de Diciembre de 2023 (~12.5% de los datos).

### 5.2 Tabla Comparativa de Modelos (Conjunto de Prueba)

| Modelo / Algoritmo | Inflow MAE | Outflow MAE | **Balance Neto MAE** | **Balance Neto RMSE** | **Balance $R^2$** | **Mejora vs Baseline** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Heurístico** (Mediana Histórica) | 0.5210 | 0.5430 | **0.6965** | **1.3967** | 0.5120 | *Referencia* |
| **Random Forest Regressor** | 0.4012 | 0.4185 | **0.5180** | **1.1420** | 0.6734 | **+25.62%** |
| **LightGBM Regressor (Optimizado)** | **0.3715** | **0.3892** | **0.4721** | **1.0746** | **0.7289** | **+32.22%** |

========================================================================================

## 6. Seguimiento con MLflow Tracking y Model Registry

Se integró **MLflow** para gobernar el ciclo de vida del modelo de forma profesional:

### 6.1 MLflow Tracking
* **Experimento:** `mibici_hourly_station_balance`
* **Parámetros Registrados:** `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `subsample`, `colsample_bytree`, `features_count`.
* **Métricas Registradas:** `inflow_mae`, `inflow_rmse`, `outflow_mae`, `outflow_rmse`, `balance_mae`, `balance_rmse`, `balance_r2`.
* **Artefactos Guardados:** Pesos de modelos (`lightgbm_inflow.joblib`, `lightgbm_outflow.joblib`), esquemas de datos y gráficas de evaluación.

### 6.2 MLflow Model Registry
Los mejores modelos fueron registrados formalmente en el catálogo de modelos de MLflow:
* **`mibici_inflow_model`** (Versión 1) $\rightarrow$ Tagged: `Candidate-Production`
* **`mibici_outflow_model`** (Versión 1) $\rightarrow$ Tagged: `Candidate-Production`

========================================================================================

## 7. Participación e Interacción de Roles del Equipo 6

| Integrante | Rol en el Proyecto | Responsabilidades y Entregables en la Fase 2 |
| :--- | :--- | :--- |
| **Leonel Matus Climaco** | **MLOps Engineer & Pipeline Architect** | Arquitectura modular Cookiecutter, diseño de `Makefile`, orquestación de Scikit-Learn Pipelines, integración de MLflow Tracking y registro de modelos en el Model Registry. |
| **Kenji Ramírez** | **Data Engineer & Feature Architect** | Refactorización de `dataset.py`, generación de la cuadrícula horaria continua de 359 estaciones, ingeniería de features cíclicas y lags en `features.py`. |
| **Iván V.** | **ML Engineer & Model Evaluator** | Entrenamiento de modelos (LightGBM, Random Forest), tuning de hiperparámetros, implementación de métricas de negocio y evaluación de balance en `evaluate.py`. |

### Interacción y Flujo de Trabajo Colaborativo:
1. **Data Engineering $\rightarrow$ ML Engineering:** Kenji generó los datasets limpios y transformados mediante `dataset.py` y `features.py`.
2. **ML Engineering $\rightarrow$ MLOps:** Iván ajustó los hiperparámetros y validó las métricas de balance neto.
3. **MLOps $\rightarrow$ Producción:** Leonel integró los transformadores en el `Pipeline` de Scikit-Learn, automatizó las corridas en MLflow y estructuró las pruebas unitarias y el `Makefile`.
4. **Control de Versiones:** Colaboración mediante ramas de feature en GitHub con revisiones de código (*Pull Requests*) hacia `feature/mlops-pipeline-2023`.

========================================================================================

## 8. Guía de Reproducibilidad Paso a Paso

Para replicar todos los experimentos y validar los resultados:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Kenjikey/ML_pipeline_eq6.git
cd ML_pipeline_eq6
git checkout feature/mlops-pipeline-2023

# 2. Instalar dependencias
make setup

# 3. Procesar datos y construir features
make data
make features

# 4. Entrenar y registrar en MLflow
make train

# 5. Ejecutar inferencia de muestra
make predict

# 6. Validar pruebas unitarias
make test

# 7. Abrir la interfaz visual de MLflow
make mlflow_ui
```
*(MLflow UI disponible en: `http://localhost:5000`)*
