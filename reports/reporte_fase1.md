# Reporte Técnico Ejecutivo - Fase 1: MLOps Pipeline MiBici (2023)
## Sistema Predictivo de Rebalanceo de Flota por Estación y Hora

**Autores:** Leonel Matus Climaco, Kenji Ramírez, Iván V.  
**Rol:** Equipo 6 - MLOps Pipeline & Machine Learning Engineering  
**Proyecto:** MiBici MLOps - Fase 1  
**Repositorio GitHub:** `https://github.com/Kenjikey/ML_pipeline_eq6.git`  
**Rama Activa:** `feature/mlops-pipeline-2023`  

---

## 1. Resumen Ejecutivo y Machine Learning Canvas

El sistema de bicicletas públicas del Área Metropolitana de Guadalajara (**MiBici**) registró **3,908,646 viajes** durante el año 2023 en una red de **359 estaciones activas**. El desbalance asimétrico entre estaciones emisoras (zonas residenciales) y receptoras (zonas corporativas y educativas) genera eventos diarios de desabasto y saturación en horas pico, provocando pérdida de viajes e insatisfacción de los usuarios.

Para resolver este desafío operacional, se diseñó un pipeline de MLOps modular y reproducible estructurado bajo el estándar Cookiecutter Data Science. El modelo predice el balance neto de bicicletas por estación-hora ($\text{Balance} = \text{Entradas} - \text{Salidas}$) desacoplando la demanda en dos regresores independientes, alcanzando un **MAE de 0.4721 bicicletas/hora** (superando a la heurística baseline en un **+32.22%**).

### Machine Learning Canvas (Fase 1 - Alineado y Enriquecido)

| Dimensión | Especificación del Pipeline |
| :--- | :--- |
| **Problema de Negocio** | Saturación de muelles y desabasto de bicicletas en estaciones críticas durante horas pico laborales. |
| **Objetivo de ML** | Predecir el balance neto de flujo ($\text{Balance} = \text{Entradas} - \text{Salidas}$) en la estación $s$ para la hora $t+1$. |
| **Formulación Matemática** | Regresión desacoplada: $\hat{E}_{s, t+1}$ (Entradas) y $\hat{S}_{s, t+1}$ (Salidas) con $\hat{B}_{s, t+1} = \hat{E}_{s, t+1} - \hat{S}_{s, t+1}$. |
| **Fuentes de Datos Internas** | - 12 datasets mensuales de datos abiertos MiBici 2023 (~3.91 millones de registros brutos).<br>- Manifiesto de linaje (`raw_v1` en Google Drive / Manifest JSON). |
| **Fuentes de Datos Externas (Nuevas)** | - **Catálogo Físico de Estaciones (API/GIS):** Coordenadas geográficas (`latitud`, `longitud`) para mapeo de calor y cálculo de distancias euclidianas/manhattan entre estaciones vecinas.<br>- **Capacidad de Muelles (`capacidad_total`):** Número total de slots de anclaje por estación para calcular el porcentaje de saturación ($\frac{\text{Bicis Físicas}}{\text{Capacidad}} \times 100\%$). |
| **Ingeniería de Características** | Componentes cíclicas ($\sin/\cos$ de hora y mes), lags temporales ($t-1\text{h}, t-2\text{h}, t-24\text{h}, t-168\text{h}$), medias móviles de 6h y flags de día laboral. |
| **Acción Operativa** | Clasificación en 3 estados combinada con capacidad física:<br>- **Déficit ($\le -2$ o Saturación $<15\%$):** Abastecimiento urgente de bicicletas.<br>- **Superávit ($\ge +2$ o Saturación $>85\%$):** Retiro de bicicletas para liberar slots.<br>- **Estable ($-2 < B < +2$):** Operación en rango óptimo. |
| **Métricas de Éxito** | - **Técnica:** MAE Balance = 0.4721 (Baseline = 0.6966, mejora +32.22%).<br>- **Operativa:** Detección anticipada de saturación y optimización de rutas de camionetas. |
| **Infraestructura MLOps** | Formato Apache Parquet, modularidad en `src/mibici/`, y tracking de experimentos en MLflow (puerto 5001 en macOS). |

---

## 2. Justificación Metodológica del Uso del Año 2023

En el análisis exploratorio multi-anual se evaluó el comportamiento histórico del sistema desde 2014, sustentando la selección de 2023 como período de entrenamiento y evaluación bajo tres criterios técnicos:

1. **Topología Dinámica de la Red (Expansión de Estaciones):**
   - En años iniciales (2014–2018), MiBici operaba con ~86 a 120 estaciones. Para 2023, la red alcanzó **359 estaciones activas**.
   - Entrenar modelos de predicción espacial con datos anteriores a 2022 introduciría distorsiones severas, obligando al modelo a aprender patrones de movilidad sobre rutas que no contaban con estaciones conectadas.
2. **Quiebre Estructural Post-Pandemia (Nueva Normalidad):**
   - Los años 2020 y 2021 presentan anomalías profundas de confinamiento y cambios radicales en patrones laborales/escolares.
   - El año 2023 refleja fielmente el régimen de movilidad estabilizado actual (esquemas híbridos de trabajo y retorno escolar presencial).
3. **Poder Estadístico y Suficiencia Muestral:**
   - 2023 aporta **3,908,646 viajes reales** y una serie temporal de **3,144,840 observaciones estación-hora**, volumen óptimo para capturar estacionalidad horaria, semanal y mensual sin requerir extrapolaciones artificiales.

---

## 3. Ingesta, Limpieza de Datos y Linaje (Data Lineage Manifest)

A través del módulo `src/mibici/dataset.py`, los 12 archivos CSV mensuales fueron procesados bajo rigurosas reglas de calidad:

1. **Filtrado de Registros Inválidos:**
   - Descarte de viajes con estaciones de origen o destino nulas (`NaN`).
   - Eliminación de micro-viajes ($<60$ segundos con mismo origen y destino), correspondientes a candados trabados o fallas mecánicas al intentar desbloquear.
   - Eliminación de viajes con duración anómala ($>24$ horas), originados por retenciones indebidas o fallas en el anclaje de destino.
2. **Construcción del Grid Temporal Continuo:**
   - Matriz continua de $359 \text{ estaciones} \times 8,760 \text{ horas del año 2023} = \mathbf{3,144,840 \text{ filas}}$.
   - Imputación determinista de $0$ en intervalos sin flujo, garantizando que el dataset intermedio (`data/interim/hourly_balance_2023.parquet`) contenga 0 valores nulos.

### Estructura del Manifiesto de Linaje Consolidado (`manifest.json`)

```json
[
  {
    "id": "raw_v1",
    "archivo": "mibici_raw_v1.tar.xz",
    "sha256": "86a97d3b37343b88e3631eca2d877287a496ffac74bd86acdc8e97e35e7931f5",
    "url": "https://drive.google.com/file/d/1qaQ0reoO8ju_6j1bpB-UxCCBjSb17g4N/view?usp=share_link",
    "origen": [],
    "transformaciones": [
      "Empaquetar y comprimir los archivos originales de datos abiertos sin modificacion"
    ],
    "commit": null
  },
  {
    "id": "hourly_balance_2023_v1",
    "archivo": "hourly_balance_2023.parquet",
    "sha256": "c71e895271e7a17ffd556959cdd1ef516f258bfb8e5940470d5526c71a99ebd7",
    "url": "PENDING_DRIVE_UPLOAD",
    "origen": ["raw_v1"],
    "transformaciones": [
      "Filtro de micro-viajes (<60s con mismo origen y destino)",
      "Filtro de anomalias de retencion (>24 horas)",
      "Agrupacion horaria de entradas y salidas por estacion",
      "Construccion de grid continuo de 8,760 horas x 359 estaciones con imputacion determinista de ceros"
    ],
    "commit": "6c82d9c"
  },
  {
    "id": "features_2023_v1",
    "archivo": "features_2023.parquet",
    "sha256": "78c986ec35f62ca7707a40a6971cbd76b0d35c4dacdb15c5e61b283765bb20a4",
    "url": "PENDING_DRIVE_UPLOAD",
    "origen": ["hourly_balance_2023_v1"],
    "transformaciones": [
      "Generacion de variables ciclicas seno/coseno de hora y mes",
      "Generacion de lags temporales (t-1h, t-2h, t-24h, t-168h)",
      "Calculo de medias moviles de 6 horas para entradas y salidas",
      "Banderas de dia laboral vs fin de semana"
    ],
    "commit": "6c82d9c"
  }
]
```

---

## 4. Hallazgos Clave del Análisis Exploratorio de Datos (EDA)

A partir de la ejecución del notebook `notebooks/1.0-eda.ipynb` y las figuras almacenadas en `reports/figures/`, se consolidaron los siguientes hallazgos:

1. **Patrón Bimodal Laboral vs. Unimodal Fin de Semana:**
   - De lunes a viernes, la demanda presenta dos picos pronunciados: **Pico Matutino (07:00–09:00)** y **Pico Vespertino (18:00–20:00)**.
   - Los fines de semana muestran una caída del ~40% en volumen total y una distribución unimodal meseta centrada al mediodía.
2. **Esparcimiento de Flujo (Sparsity):**
   - El **53.15%** de los intervalos estación-hora registran 0 entradas y 0 salidas, justificando el desacoplamiento en dos modelos regresores en lugar de predecir balance directamente con funciones de pérdida cuadráticas que colapsan hacia la media.
3. **Concentración Extrema en Nodos Críticos:**
   - La **Estación 51** es el nodo más crítico de la red, acumulando **70,200 unidades de desbalance absoluto anual**, con un superávit neto de $+18,814$ bicicletas.
   - La Estación 51 exhibe una dinámica pendular extrema: salidas dominantes a las 08:00 hrs (déficit matutino) y llegadas masivas a las 18:00 hrs (superávit vespertino).

---

## 5. Arquitectura del Pipeline y Resultados de Modelado

```text
[12 CSVs Raw 2023 (3.91M viajes)]
              │
              ▼  (src/mibici/dataset.py)
[data/interim/hourly_balance_2023.parquet (3.14M filas x 6 col)]
              │
              ▼  (src/mibici/features.py)
[data/processed/features_2023.parquet (3.14M filas x 33 col)]
              │
              ▼  (src/mibici/modeling/train.py) -> HistGradientBoosting (E y S) + MLflow
[data/processed/test_predictions_2023.csv (Métricas y Alertas Operativas)]
```

### Tabla Comparativa de Rendimiento

| Métrica | Heurística Baseline (Media Histórica) | Pipeline ML (HistGradientBoosting) | Mejora Relativa |
| :--- | :---: | :---: | :---: |
| **MAE Entradas ($\hat{E}$)** | 0.6288 bicis/h | **0.4285 bicis/h** | **+31.85%** |
| **MAE Salidas ($\hat{S}$)** | 0.6288 bicis/h | **0.4285 bicis/h** | **+31.85%** |
| **MAE Balance Neto ($\hat{B}$)** | 0.6966 bicis/h | **0.4721 bicis/h** | **+32.22%** |

---

## 6. Propuesta de Mejora para Fases Futuras: Detección de Bicicletas Defectuosas e Inventario Fantasma (*Ghost Inventory*)

### Justificación de Negocio
En la operación física, una estación puede reportar 3 bicicletas en muelle, pero si 2 presentan averías mecánicas (llanta ponchada, cadena rota o frenos dañados), la **capacidad real disponible es únicamente de 1 bicicleta**, detonando fallas de servicio no anticipadas.

### Arquitectura de la Mejora
En lugar de descartar los micro-viajes durante la fase de limpieza, se propone canalizarlos a un subsistema de monitoreo de salud de flota (`src/mibici/maintenance.py`):

1. **Inferencia de Falla por Rechazos Consecutivos:**
   - Si una unidad (`Bicicleta_Id`) registra $\ge 2$ micro-viajes consecutivos ($<90$ segundos con mismo origen y destino) sin viajes exitosos intermedios, se etiqueta automáticamente como `DEFECTUOSA_CONFIRMADA`.
2. **Inferencia por Anomalía de Estancamiento (*Dwell Time*):**
   - Detección de unidades que permanecen más de 24 horas continuas en estaciones de alta rotación (como Estación 51) sin ser tomadas por los usuarios.
3. **Cálculo de Inventario Útil Real:**
   $$\text{Inventario Útil} = \text{Bicicletas Físicas} - \text{Bicicletas con Alerta de Avería}$$
   Esta métrica se integrará con el modelo predictivo para despachar de forma simultánea camiones de rebalanceo y cuadrillas de mantenimiento mecánico.

---

## 7. Conclusiones y Estado del Proyecto
La Fase 1 queda completada, validada y versionada en Git, cumpliendo con los estándares de reproducibilidad, modularidad, persistencia eficiente en Parquet, linaje de datos mediante manifiestos e integración con el Machine Learning Canvas del equipo.
