# MiBici — Pipeline de preparación de datos

Proyecto del **Equipo 6 · Operaciones de aprendizaje automático (Grupo 10)** para analizar y anticipar el balance de entradas y salidas por estación del sistema de bicicletas públicas MiBici.

El repositorio contiene el flujo de la **fase 1** para integrar los archivos mensuales de viajes de 2023, validar y limpiar los registros, construir el balance horario por estación y preparar variables temporales para etapas posteriores de modelado.

## Contenido

- [Alcance del flujo](#alcance-del-flujo)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Preparar los datos](#preparar-los-datos)
- [Ejecutar el pipeline](#ejecutar-el-pipeline)
- [Reglas de calidad y transformaciones](#reglas-de-calidad-y-transformaciones)
- [Ejecución verificada](#ejecución-verificada)
- [Pruebas](#pruebas)
- [Trazabilidad y reproducibilidad](#trazabilidad-y-reproducibilidad)
- [Limitaciones conocidas](#limitaciones-conocidas)

## Alcance del flujo

El flujo implementado y documentado en esta fase se ejecuta en dos etapas:

| Etapa | Módulo | Responsabilidad |
|---|---|---|
| 1. Integración y calidad | `mibici.dataset` | Lee los archivos mensuales, valida los registros, aplica reglas de limpieza, genera el balance horario y registra la ejecución y sus reportes en MLflow. |
| 2. Ingeniería de características | `mibici.features` | Construye variables de calendario, codificaciones cíclicas, retardos y promedios históricos por estación; valida la continuidad temporal y evita usar información de la hora objetivo. |

Esta fase prepara conjuntos de datos trazables para experimentos posteriores. La ejecución verificada descrita aquí corresponde a `dataset` y `features`; no debe interpretarse como evidencia de un modelo predictivo ya entrenado o desplegado.

## Estructura del repositorio

La estructura publicada en la rama principal es:

```text
ML_pipeline_eq6/
├── docs/             # Documentación del proyecto
├── manifests/        # Manifiestos de versiones de datos
├── mibici/           # Código fuente del pipeline
├── models/           # Directorio para artefactos relacionados con modelos
├── notebooks/        # Notebooks de exploración y análisis
├── references/       # Material de referencia del proyecto
├── reports/          # Reportes generados por los procesos
├── tests/            # Pruebas automatizadas
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.cfg
```

Los datos de entrada, las salidas generadas, la base de datos local de MLflow y sus artefactos no se versionan en Git. Las carpetas correspondientes se crean localmente cuando se ejecuta el flujo.

## Requisitos e instalación

Se requiere Python y Git. La versión de Python y las dependencias del proyecto deben respetar la configuración de `pyproject.toml` y `requirements.txt`.

1. Clonar el repositorio y entrar al directorio:

   ```bash
   git clone https://github.com/Kenjikey/ML_pipeline_eq6.git
   cd ML_pipeline_eq6
   ```

2. Crear y activar un entorno virtual:

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux o macOS:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar las dependencias:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Comprobar que las pruebas pasan:

   ```bash
   python -m pytest tests -q
   ```

## Preparar los datos

Coloca en `data/raw/` los **12 archivos CSV originales de 2023**, con estos nombres:
Los archivos se encuentran temporalmente en la siguiente liga: https://drive.google.com/open?id=1sk2UJWuqsHCMgpNiRqWmweCJZiwPpq98&usp=drive_fs

```text
datos_abiertos_2023_01.csv
datos_abiertos_2023_02.csv
datos_abiertos_2023_03.csv
datos_abiertos_2023_04.csv
datos_abiertos_2023_05.csv
datos_abiertos_2023_06.csv
datos_abiertos_2023_07.csv
datos_abiertos_2023_08.csv
datos_abiertos_2023_09.csv
datos_abiertos_2023_10.csv
datos_abiertos_2023_11.csv
datos_abiertos_2023_12.csv
```

El flujo espera los 12 meses. Si falta un archivo, el proceso se detiene para evitar tratar un mes no disponible como si tuviera cero viajes. Los datos originales no están incluidos en el repositorio: deben estar disponibles localmente antes de ejecutar las etapas.

## Ejecutar el pipeline

Los comandos siguientes se ejecutan desde la raíz del repositorio y con el entorno virtual activo.

### 1. Iniciar MLflow

En una terminal, inicia el servidor local y déjalo ejecutándose:

```bash
mlflow ui \
  --host 127.0.0.1 \
  --port 5001 \
  --workers 1 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts
```

En Windows PowerShell puedes ejecutar el mismo comando en una sola línea:

```powershell
mlflow ui --host 127.0.0.1 --port 5001 --workers 1 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

### 2. Ejecutar las etapas

En otra terminal, con el mismo entorno virtual activo:

```bash
python -m mibici.dataset
python -m mibici.features
```

### 3. Consultar los resultados

Abre [http://127.0.0.1:5001](http://127.0.0.1:5001) y consulta el experimento `mibici_balance_estaciones_2023`. Una ejecución correcta genera dos runs finalizados:

- `dataset`: integración, limpieza, agregación y reporte de calidad.
- `features`: preparación de las variables y reporte de calidad correspondiente.

Los reportes locales se guardan en `reports/quality/` e incluyen el identificador de la ejecución en su nombre.

El servidor utiliza el puerto 5001 por defecto. Para utilizar otro servidor de seguimiento, configura `MLFLOW_TRACKING_URI` o proporciona `--tracking-uri` a cada etapa, según corresponda.

## Reglas de calidad y transformaciones

### Limpieza de viajes

- Los identificadores deben ser enteros positivos y las fechas deben ser válidas.
- La fecha y hora de finalización deben ser posteriores a las de inicio.
- Cada registro descartado se contabiliza una sola vez.
- Se conservan los viajes cuyo inicio y fin ocurren dentro de 2023.
- Los datos demográficos faltantes no provocan que se descarte un viaje.

### Duplicados y conflictos

- La revisión de duplicados se realiza considerando conjuntamente todos los archivos mensuales.
- Se comparan el identificador del viaje, las estaciones y las fechas.
- Si un mismo identificador representa movimientos distintos, el proceso se detiene para que se revise la fuente, en lugar de elegir arbitrariamente un registro.

### Detección de valores atípicos

Se reportan, entre otros, los siguientes casos:

- Duraciones fuera de los límites de `Q1 − 1.5 × IQR` y `Q3 + 1.5 × IQR`.
- Viajes de menos de 60 segundos que inician y terminan en la misma estación.
- Viajes con duración superior a 24 horas.

Los límites IQR se utilizan para diagnóstico; no constituyen por sí mismos una regla para eliminar observaciones del conjunto. En la ejecución verificada, las duraciones atípicas se conservaron.

### Balance horario por estación

- Una **salida** se cuenta en la hora de inicio del viaje.
- Una **entrada** se cuenta en la hora de finalización.
- El balance se calcula como `entradas - salidas`.
- Se verifica que los totales de entradas y salidas coincidan con el número de viajes limpios.

La frecuencia horaria corresponde a este experimento y no define necesariamente el horizonte final del producto ni representa directamente la disponibilidad de bicicletas en una estación.

### Ceros y estaciones sin movimientos

Un cero en entradas o salidas significa que no se registraron movimientos de ese tipo en el periodo. Un balance igual a cero significa que las entradas y salidas fueron iguales, no que no hubo actividad. Con los datos disponibles tampoco se identifican necesariamente los periodos en que una estación estuvo cerrada, fue dada de alta o dejó de operar.

### Ingeniería de características

La preparación incluye:

- Variables de calendario.
- Codificación cíclica de hora y día de la semana.
- Retardos de 1, 2, 24 y 168 horas.
- Promedios de las 3 y 6 horas anteriores por estación.
- Validación de continuidad horaria.

Se eliminan las primeras 168 horas de cada serie porque no cuentan con historial suficiente. Los promedios utilizan únicamente horas anteriores a la hora que se busca caracterizar, para evitar fuga de información temporal.

La bandera de hora pico conserva la hipótesis del flujo original (7–9 y 18–20); debe contrastarse con el análisis exploratorio de datos antes de asumir que representa los patrones definitivos de demanda.

## Ejecución verificada

La ejecución documentada se realizó el **25 de octubre de 2026**, utilizando los 12 CSV de 2023 extraídos del archivo original `mibici_raw_v1.tar.xz` y MLflow local en el puerto 5001.

| Indicador | Resultado |
|---|---:|
| Viajes antes de limpiar | 4,293,115 |
| Descartados por duración no positiva | 17,993 |
| Descartados por cruzar el año | 5 |
| Viajes conservados | 4,275,117 |
| Estaciones identificadas | 359 |
| Filas del balance horario | 3,144,840 |
| Filas preparadas | 3,084,528 |
| Características totales | 33 |
| Filas excluidas por falta de historial de 168 horas | 60,312 |
| Valores nulos en las características finales | 0 |

Las ejecuciones finalizadas de MLflow registradas para esa corrida fueron:

| Etapa | `run_id` |
|---|---|
| Limpieza y agregación (`dataset`) | `80f4168ba16e4edd8ffb474279fceb20` |
| Preparación (`features`) | `4a7b9060d1c3435aa21d4d4a85a74de1` |

También se identificaron **306 combinaciones de estación y mes sin movimientos registrados** y **89,758 duraciones atípicas según IQR**, que se conservaron conforme a las reglas descritas. Los reportes de `reports/quality/` contienen el detalle y las huellas correspondientes.

Estos valores son evidencia de una ejecución concreta; pueden cambiar al regenerar el pipeline con otras entradas, reglas o versiones del código.

## Pruebas

Ejecuta desde la raíz del repositorio:

```bash
python -m pytest tests -q
```

Las pruebas generan datos sintéticos temporales y utilizan una base de MLflow aislada. Por ello, permiten comprobar el comportamiento del código sin necesitar los archivos reales de MiBici ni modificar la base de seguimiento utilizada para los experimentos locales.

## Trazabilidad y reproducibilidad

El pipeline registra en MLflow las reglas aplicadas, conteos, metadatos de los conjuntos de datos, fuentes del código, entorno de ejecución y reportes de calidad. Las rutas, las huellas SHA-256 y el `run_id` ayudan a relacionar las salidas de ambas etapas.

La huella que muestra MLflow no debe confundirse con el SHA-256 del archivo: son identificadores distintos y se conservan ambos.

Consideraciones para reproducir o publicar una nueva versión:

1. Usa los archivos originales correspondientes y conserva sus nombres y procedencia.
2. No reutilices hashes de manifiestos anteriores para salidas regeneradas con reglas distintas. Los archivos de `manifests/` describen versiones compartidas anteriormente.
3. Después de ejecutar el flujo con los datos reales, registra la nueva versión, su SHA-256, entradas, transformaciones, ubicación y commit del código correspondiente.
4. Revisa los cambios del repositorio antes de ejecutar: cuando hay cambios sin confirmar, MLflow registra esa condición y adjunta los archivos fuente utilizados.
5. Conserva los reportes de calidad como evidencia complementaria; no sustituyen al manifiesto de datos compartido.

Los datos, la base SQLite de MLflow y los artefactos locales se mantienen fuera de Git, de acuerdo con las reglas de `.gitignore`.

## Limitaciones conocidas

- La ejecución necesita los 12 archivos mensuales completos de 2023.
- Las variables históricas presuponen que las horas previas ya fueron observadas. Por sí solas no implementan predicciones recursivas de varios pasos sin observaciones futuras.
- La frecuencia horaria es una decisión de este experimento y debe validarse contra el objetivo final de predicción.
- La definición de hora pico es una hipótesis heredada y requiere validación con EDA.
- La estación se conserva como identificador numérico en esta preparación; debe revisarse si esa representación es adecuada para el modelo que se elija.
- Los ceros observados no permiten distinguir por sí solos entre ausencia de demanda y periodos sin operación de una estación.

---

**Equipo 6 · Operaciones de aprendizaje automático · Grupo 10**
