# Reporte Técnico Ejecutivo - Fase 1: MLOps Pipeline MiBici (2023)
## Sistema Predictivo de Rebalanceo de Flota por Estación y Hora


**Proyecto:** MiBici MLOps - Fase 1  
**Repositorio:** `https://github.com/Kenjikey/ML_pipeline_eq6.git`  
**Rama:** `feature/mlops-pipeline-2023`  

---

## 1. Resumen Ejecutivo y Machine Learning Canvas

El sistema de bicicletas públicas del Área Metropolitana de Guadalajara (**MiBici**) registró **3,908,646 viajes** durante el año 2023 en una red de **359 estaciones activas**. El desbalance asimétrico entre estaciones emisoras (zonas residenciales) y receptoras (zonas corporativas y educativas) genera eventos diarios de desabasto y saturación en horas pico, provocando pérdida de viajes e insatisfacción de los usuarios.

Para resolver este desafío operacional, se diseñó un pipeline de MLOps modular y reproducible estructurado bajo el estándar Cookiecutter Data Science. El modelo predice el balance neto de bicicletas por estación-hora ($\text{Balance} = \text{Entradas} - \text{Salidas}$) desacoplando la demanda en dos regresores independientes, alcanzando un **MAE de 0.4721 bicicletas/hora** (superando a la heurística baseline en un **+32.22%**).

### Machine Learning Canvas (Fase 1)



---

## 2. Ingesta, Limpieza de Datos y Construcción del Grid Continuo

A través del módulo `src/mibici/dataset.py`, los 12 archivos CSV mensuales fueron procesados bajo rigurosas reglas de calidad de datos:

1. **Filtrado de Registros Inválidos:**
   - Descarte de viajes con estaciones de origen o destino nulas (`NaN`).
   - Eliminación de micro-viajes ($<60$ segundos con mismo origen y destino), correspondientes a candados trabados o fallas mecánicas inmediatas al intentar desbloquear la bicicleta.
   - Eliminación de viajes con duración anómala ($>24$ horas), originados por retenciones indebidas o fallas en el anclaje de destino.
2. **Construcción del Grid Temporal Continuo:**
   - Para evitar huecos temporales en horas de baja demanda o madrugada, se construyó una matriz continua de $359 \text{ estaciones} \times 8,760 \text{ horas del año 2023} = \mathbf{3,144,840 \text{ filas}}$.
   - Imputación determinista de $0$ en intervalos sin flujo, garantizando que el dataset intermedio (`data/interim/hourly_balance_2023.parquet`) contenga 0 valores nulos.

---

## 3. Hallazgos Clave del Análisis Exploratorio de Datos (EDA)

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

## 4. Arquitectura del Pipeline y Resultados de Modelado

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

## 5. Propuesta de Mejora para Fases Futuras: Detección de Bicicletas Defectuosas e Inventario Fantasma (*Ghost Inventory*)

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

## 6. Conclusiones y Estado del Proyecto
La Fase 1 queda completada, validada y versionada en Git, cumpliendo con los estándares de reproducibilidad, modularidad, persistencia eficiente en Parquet y visualización interactiva.