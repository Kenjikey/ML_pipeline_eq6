# MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Anticipar el balance de entradas y salidas por estacion de MiBici

## Flujo de datos de fase 1

Flujo base de carga, preparación, ajustes de calidad y trazabilidad.

### Ejecución local

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Colocar los 12 archivos originales de 2023 directamente en `data/raw/`, con nombres
`datos_abiertos_2023_01.csv` a `datos_abiertos_2023_12.csv`. Si falta alguno, el flujo se
detiene para evitar representar un mes no disponible mediante ceros.

En una terminal iniciar MLflow y mantenerla abierta:

```bash
mlflow ui --host 127.0.0.1 --port 5001 --workers 1 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

En otra terminal, activar el mismo entorno virtual y ejecutar:

```bash
python -m mibici.dataset
python -m mibici.features
```

Abrir <http://127.0.0.1:5001>. En el experimento `mibici_balance_estaciones_2023` deben
aparecer dos ejecuciones finalizadas: `dataset` y `features`. Cada una registra reglas,
conteos, metadatos de los datasets, fuentes del código, entorno y un reporte de calidad.
Los reportes locales quedan en `reports/quality/`, con el identificador de la ejecución
en el nombre. Las rutas, SHA-256 y el `run_id` permiten relacionar las etapas.
La huella que muestra MLflow es distinta del SHA-256 del archivo; se conservan ambas.

El servidor predeterminado usa el puerto 5001. Para otro servidor, definir
`MLFLOW_TRACKING_URI` o pasar `--tracking-uri` a cada etapa. Los datos, la base de MLflow
y sus artefactos permanecen locales y están excluidos de Git.

### Reglas y justificación

- **Limpieza:** se exigen identificadores enteros positivos y fechas válidas, con fin
  posterior al inicio. Se cuenta cada descarte una sola vez. Se conservan los viajes
  cuyo inicio y fin caen en 2023.
- **Duplicados:** se comparan identificador, estaciones y fechas en todos los archivos
  juntos. Si el mismo identificador corresponde a movimientos distintos, se detiene el
  proceso para revisar la fuente. Los datos demográficos faltantes no eliminan viajes.
- **Atípicos:** se reportan duraciones fuera de Q1 − 1.5 × IQR y Q3 + 1.5 × IQR, viajes
  de menos de 60 segundos con la misma estación y viajes de más de 24 horas.
  Los límites IQR se calculan solo para el diagnóstico, no se utilizan para entrenar
  ni filtrar el conjunto de prueba.
- **Agregación:** se cuentan entradas al finalizar el viaje y salidas al iniciarlo.
  El balance es entradas menos salidas. Se comprueba que ambos totales coincidan con
  los viajes limpios. La frecuencia de una hora es la opción de este experimento;
  no representa un horizonte definitivo del producto ni disponibilidad de bicicletas.
- **Ceros:** en entradas y salidas indican ausencia de movimientos registrados de ese tipo.
  En el balance, indican igualdad entre entradas y salidas, aunque haya habido movimientos.
  No conocemos periodos de cierre, alta o baja de las estaciones.
- **Preparación:** calendario, codificación cíclica de hora y día de semana, retardos de
  1, 2, 24 y 168 horas y medias de las 3 y 6 horas anteriores por estación. Se comprueba
  continuidad horaria y se eliminan las primeras 168 horas de cada serie por falta de
  historial. Ninguna media incluye la hora que se intenta predecir. La bandera de hora
  pico conserva la hipótesis del flujo original (7–9 y 18–20), esto requiere contrastarse en EDA.
- **Escalado y PCA:** se conservan las unidades de conteo y la interpretación de las
  variables. La estación se conserva como identificador numérico en ese modelo, lo que
  constituye una decisión provisional.

Las variables históricas presuponen que las horas anteriores ya se observaron. Este
flujo no implementa por sí solo predicciones de varios pasos sin observaciones futuras.

### Versiones y evidencia

Los archivos de `manifests/` describen versiones anteriores compartidas en Drive.
No deben reutilizarse sus hashes para salidas regeneradas con reglas diferentes.
Después de ejecutar este flujo con los datos reales, registrar la nueva versión con
su SHA-256, entradas, transformaciones, ubicación y commit del código correspondiente.
Mientras haya cambios sin confirmar, MLflow registra esa condición y adjunta los fuentes
exactos usados. Los reportes de calidad no sustituyen al manifiesto compartido.

Las pruebas generan datos sintéticos temporales y una base MLflow aislada. Sirven para
comprobar el código.

### Ejecución verificada del 25 de octubre de 2026

Se extrajeron los 12 CSV de 2023 del archivo original `mibici_raw_v1.tar.xz` y se ejecutaron
las dos etapas contra MLflow local en el puerto 5001.

| Resultado | Cantidad |
| --- | ---: |
| Viajes antes de limpiar | 4,293,115 |
| Descartados por duración no positiva | 17,993 |
| Descartados porque cruzan el año | 5 |
| Viajes conservados | 4,275,117 |
| Estaciones | 359 |
| Filas del balance horario | 3,144,840 |
| Filas preparadas / columnas totales | 3,084,528 / 33 |
| Filas sin historial de 168 horas | 60,312 |
| Nulos en las características finales | 0 |

Los reportes de `reports/quality/` contienen las huellas completas y los motivos de
descarte. Las ejecuciones finalizadas son:

- Limpieza y agregación: `80f4168ba16e4edd8ffb474279fceb20`.
- Preparación: `4a7b9060d1c3435aa21d4d4a85a74de1`.

Se detectaron 306 combinaciones de estación y mes sin movimientos registrados. Además,
89,758 duraciones fueron atípicas según IQR y se conservaron según la regla documentada.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

