# Proyecto MiBici · Equipo 27

**TC5061 · Proyecto integrador · Entregas 1 y 2**

Propuesta y base reproducible de Machine Learning para analizar la demanda, el
desequilibrio de flujo y los patrones de uso de estaciones de MiBici, el sistema
de bicicleta pública del Área Metropolitana de Guadalajara.

La implementación procesó **4,497,119 viajes válidos**, construyó un panel de
**4,125,960 filas estación × hora** y registró cuatro runs de referencia en MLflow.
La ingesta se ejecutó dos veces con la misma versión de datos.

La estructura se generó con Cookiecutter Data Science y su módulo se llama `mibici`.

## Resultados de la segunda entrega

- [Reporte PDF de partes B y C](output/pdf/TC5061_Reporte_MLFlow_Equipo27.pdf).
- [Reporte editable con tablas y evidencia](tec-mlops-equipo-27/reports/reporte_mlflow.md).
- [Resultados de experimentación](tec-mlops-equipo-27/reports/experiment_results.json).
- [Registry y verificación de carga por alias](tec-mlops-equipo-27/reports/registry_results.json).
- [Validación ejecutada y límites](tec-mlops-equipo-27/references/validacion-entrega2.md).

La parte A está integrada en `main` mediante el PR #10. La parte C se integró
en `feat/mlflow-experiments` mediante el PR #12; la incorporación de B y C a
`main` sigue pendiente en el estado revisado el 25 de octubre de 2026.
Los tres modelos `champion` se verificaron en el servidor MLflow de la ejecución
documentada; sus artefactos y base de tracking no se incluyen en Git.

## Documentos y evidencias

- [Arquitectura, pipeline, modelos candidatos y justificación](tec-mlops-equipo-27/references/arquitectura.md).
- [Diagrama de arquitectura](tec-mlops-equipo-27/references/arquitectura.svg).
- [Manifiesto de datos](tec-mlops-equipo-27/references/data_manifest.json).
- [Resultados exactos de los baselines](tec-mlops-equipo-27/reports/baseline_results.json).
- [Interpretación de resultados y limitaciones](tec-mlops-equipo-27/reports/resultados_base.md).
- [Guía de ejecución y evidencias](tec-mlops-equipo-27/references/ejecucion-y-evidencias.md).
- [Notebook de exploración](tec-mlops-equipo-27/notebooks/1.0-equipo27-exploracion-mibici.ipynb).
- [ML Canvas](tec-mlops-equipo-27/references/ml_canvas.pdf).

## Propósito y preguntas

El objetivo es preparar y explorar millones de registros, implementar variables
reutilizables, versionar los datos y establecer baselines que permitan comparar
las mejoras de las siguientes entregas.

| Pregunta | Formulación y granularidad | Usuario y decisión |
| --- | --- | --- |
| **P1 · Regresión** | Al inicio de la hora `t`, predecir cuántos viajes comenzarán en cada estación durante esa hora. Entidad: estación × hora. | Operación: apoyar la planeación de bicicletas según demanda prevista. |
| **P2 · Clasificación** | Al inicio de `t`, predecir si la magnitud del flujo neto total de las tres horas que comienzan en `t` será al menos el 50 % de la capacidad. Entidad: estación × hora. | Rebalanceo: priorizar estaciones por probabilidad de desequilibrio. |
| **P3 · Clustering** | Agrupar estaciones por distribución horaria de uso, volumen y balance llegadas/salidas, utilizando únicamente entrenamiento. Entidad: estación. | Operación y planeación: identificar patrones similares y formular hipótesis de estrategias diferenciadas. |

**P2 es una aproximación:** MiBici no publica inventario de bicicletas por estación.
La etiqueta se calcula como `abs(suma(llegadas - salidas)) >= 0.5 × capacidad`
sobre `[t, t + 3 horas)`. No observa estaciones vacías o llenas, no detecta todos
los cruces intermedios del umbral y no permite afirmar ahorros operativos.

## Equipo y colaboración

| Integrante | Matrícula | GitHub | Responsabilidad |
| --- | --- | --- | --- |
| Humberto Alejandro Espinola González | A01840066 | hespinola | Integración, ingesta reproducible, baselines, pruebas y documentación. |
| Daniela Montserrat Enríquez Ballesteros | A01797865 | danimeb | Configuración, dependencias y módulo de feature engineering. |
| Humberto Carrillo Gómez | A01377246 | humbertCG | Preprocesamiento exploratorio. |
| Oscar Alejandro Torres Maya | A01377686 | OscarAlejandro24 | Dataset exploratorio. |

## Estructura real del repositorio

El proyecto generado está **un nivel dentro del repositorio**. Los comandos de
pruebas Python se ejecutan desde ese subdirectorio. Los comandos `mlflow run`
y el servidor de tracking de esta guía se ejecutan desde la raíz Git, donde
está `MLproject`.

```text
tec-mlops-equipo-27/                 # Raíz del repositorio Git
├── README.md                       # Este documento
├── MLproject                       # ingest, train, experiment y register
├── python_env.yaml                 # Entorno del proyecto (mismas versiones fijas)
├── .gitignore
└── tec-mlops-equipo-27/             # Proyecto generado con ccds
    ├── README.md                   # Acceso a documentación y ejecución
    ├── LICENSE
    ├── Makefile
    ├── pyproject.toml
    ├── requirements.txt
    ├── .env.example
    ├── data/                       # Contenido local excluido de Git
    │   ├── raw/                    # CSV originales
    │   └── processed/              # Parquet limpio y panel horario
    ├── mibici/
    │   ├── config.py               # Rutas, fechas, fuentes, semillas y parámetros
    │   ├── dataset.py              # Descarga, limpieza, Parquet y manifiesto
    │   ├── preparar.py             # Agregación por bloques y construcción del panel
    │   ├── features.py             # Funciones reutilizables por pregunta
    │   └── modeling/
    │       ├── train.py            # Baselines, evaluación y MLflow
    │       ├── candidates.py       # Candidatas, modelos y diagnósticos
    │       ├── experiment.py       # Baselines y búsquedas anidadas
    │       └── registry.py         # Selección, registro y alias champion
    ├── notebooks/                  # Exploración
    ├── references/                 # Manifiesto, arquitectura y guía de evidencias
    ├── reports/
    │   ├── baseline_results.json   # Métricas e IDs de runs
    │   ├── experiment_results.json # Baselines y nueve candidatas
    │   ├── registry_results.json   # Modelos, linaje y verificación por alias
    │   ├── reporte_mlflow.md       # Reporte de partes B y C
    │   ├── resultados_base.md      # Interpretación
    │   └── figures/                # Gráficas y capturas seleccionadas
    ├── models/                    # Modelos locales, excluidos de Git
    └── tests/                     # Pruebas sin descargas de MiBici
```

La plantilla también conserva archivos auxiliares y carpetas para documentación.
`predict.py` y `plots.py` no forman parte del pipeline validado de esta entrega.

## Instalación

Requisitos: **Python 3.13**, Git, acceso HTTPS a MiBici y un entorno virtual.
Se incluyen ejemplos para macOS/Linux y Windows PowerShell. Se verificó `pip check` sin conflictos
en el entorno de ejecución del equipo; las dependencias principales tienen
versiones fijas en `requirements.txt`.

```bash
git clone https://github.com/hespinola/TC5061-Proyecto-Equipo27.git tec-mlops-equipo-27
cd tec-mlops-equipo-27
# Mientras B y C no estén integradas en main:
git switch feat/mlflow-experiments

python3.13 -m venv .venv
source .venv/bin/activate
cd tec-mlops-equipo-27
python -m pip install -r requirements.txt
python -m pip check
cd ..
```

En Windows PowerShell, desde la carpeta donde se alojará el clon:

```powershell
git clone https://github.com/hespinola/TC5061-Proyecto-Equipo27.git tec-mlops-equipo-27
Set-Location tec-mlops-equipo-27
# Mientras B y C no estén integradas en main:
git switch feat/mlflow-experiments
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-Location tec-mlops-equipo-27
python -m pip install -r requirements.txt
python -m pip check
Set-Location ..
```

Ambas secuencias terminan en la raíz Git con el entorno activado. Después de
integrar B y C en `main`, se puede omitir el cambio a la rama de experimentación.

La configuración local de tracking se puede personalizar mediante `.env`, siguiendo
`.env.example`. Por defecto, `MLFLOW_TRACKING_URI` apunta al servidor local
`http://127.0.0.1:5000` (ver [MLflow Project](#mlflow-project-y-servidor-de-tracking)).

## Pruebas y ejecución reproducible

Para ejecutar las pruebas desde la raíz Git con el entorno activado:

```bash
cd tec-mlops-equipo-27
python -m unittest discover -s tests -v
cd ..
```

Para reproducir B y C, seguir las tres primeras secciones de la guía MLflow:
iniciar el servidor, configurar el entorno y ejecutar `ingest`, `experiment`
y `register`, en ese orden. La experimentación verifica los Parquet y reconstruye
el panel antes de evaluar; no requiere generar las variables manualmente.
El entrenamiento exige que el código esté confirmado en Git; si hay cambios en
`mibici/`, `requirements.txt` o `pyproject.toml`, se detiene para evitar etiquetar
un run con un commit que no describe su implementación.

La ingesta y los baselines se ejecutan sin descargas manuales, ediciones de CSV ni
dependencia de un notebook. Repetir los baselines crea nuevos IDs de run y fechas;
se comparan versión de datos, configuración y métricas, no esos identificadores.

## MLflow Project y servidor de tracking

A partir de la segunda entrega, **todas las corridas se lanzan con `mlflow run`**
desde la raíz del repositorio, no desde un notebook.

### 1. Servidor de tracking con base de datos

El Model Registry requiere un almacenamiento respaldado por base de datos. En una
terminal aparte, desde la raíz Git (la carpeta que contiene `MLproject`),
con el mismo entorno virtual activado:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

La UI queda en [http://127.0.0.1:5000](http://127.0.0.1:5000). `mlflow.db`,
`mlruns/` y `mlartifacts/` están en `.gitignore` y no se versionan.
Mantener esa terminal abierta. La ruta SQLite es relativa al directorio desde
el que se inicia el servidor; usar siempre la misma ubicación para consultar
las corridas y modelos de esa ejecución.

En la terminal donde se ejecutan las corridas:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo27
```

En PowerShell, usar estas asignaciones en lugar de `export`:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "mibici-equipo27"
```

`MLFLOW_EXPERIMENT_NAME` hace que `mlflow run` cree cada corrida en el experimento
del equipo (equivale a `--experiment-name mibici-equipo27`). Si falta, `train` se
detiene con un mensaje en lugar de registrar en `Default`.

### 2. Entorno del proyecto

`MLproject` declara `python_env.yaml` (Python 3.13 y las mismas versiones fijas de
`requirements.txt`; `tests/test_mlproject.py` falla si dejan de coincidir). MLflow
crea ese entorno de forma aislada con `--env-manager uv` (requiere `pip install uv`)
o `--env-manager virtualenv` (requiere `pyenv`). Con `--env-manager local` se usa el
entorno virtual activo, útil durante el desarrollo.

### 3. Comandos desde un clon limpio

Desde la raíz del repositorio, con el entorno instalado activado, las variables
anteriores definidas y el servidor activo (los comandos sirven en Bash y PowerShell):

```bash
# Ingesta: descarga, verifica el manifiesto y lo registra como artefacto data/.
mlflow run . -e ingest --env-manager local

# Cuatro baselines y nueve candidatas, con búsquedas anidadas.
mlflow run . -e experiment --env-manager local

# Selección y registro de los tres modelos con alias champion.
mlflow run . -e register --env-manager local
```

El entrenamiento exige el código confirmado en Git (incluidos `MLproject` y
`python_env.yaml`). Los comandos individuales de los modelos elegidos están en
[la parte C](#entrega-2--parte-c-selección-y-model-registry). Ejecutar `train`
individualmente no sustituye el resumen completo que necesita `register`.

### 4. Parámetros del punto de entrada `train`

| Parámetro | Valor por defecto | Uso |
| --- | --- | --- |
| `question` | `all` | `p1`, `p2`, `p3` o `all` (los cuatro baselines). |
| `model` | `config` → `baseline` | Modelo registrado en `config.HIPERPARAMETROS`. |
| `stage_of_work` | `config` → `baseline` o `candidate` | Etiqueta para filtrar: `baseline`, `search`, `candidate`. |
| `rezago_horas` | `config` → 168 | P1 baseline: 24 o 168. |
| `n_clusters` | `config` → `None` | P3: k fijo; `none` elige k por silhouette. |
| `n_estimators`, `max_depth`, `learning_rate` | `config` | Iteraciones, profundidad y tasa de aprendizaje de HGB en P1/P2. |
| `n_components` | `config` → 8 | Componentes de PCA en P3 con `model=pca_kmeans`. |

El valor `config` es un centinela: `MLproject` no puede importar Python, así que
`train.py` lo sustituye por el valor de `mibici/config.py`. Así, rutas, fechas,
semillas e hiperparámetros por defecto se definen en un solo lugar. Un
hiperparámetro que no aplica al modelo elegido es un error, para no registrar
corridas con valores ignorados.

En la UI, los parámetros sin prefijo (`n_clusters = config`) muestran lo que se
pidió; los de prefijo `hp_` (`hp_n_clusters = 4`) son los valores efectivos.

### 5. Etiquetas y artefactos de cada corrida

Etiquetas: `equipo`, `question`, `pregunta`, `model`, `model_family`,
`stage_of_work`, `git_commit` y `data_version`; MLflow agrega
`mlflow.source.git.commit` y el punto de entrada.

| Carpeta | Contenido |
| --- | --- |
| `config/` | `run_config.json` (pregunta, modelo, hiperparámetros, partición, semilla) y `entorno.json` (versiones). |
| `data/` | `data_manifest.json`: hashes, conteos y fuentes; nunca viajes ni datos personales. |
| `plots/` | Gráficas de desempeño de la tarea. |
| `results/` | Resumen y selección de umbral o de grupos. |

El conjunto de entrenamiento se registra con `mlflow.log_input` (esquema y digest,
sin subir filas).

### 6. Cómo agregar un modelo candidato

1. En `mibici/config.py`, agregar el modelo y sus hiperparámetros por defecto a
   `HIPERPARAMETROS[pregunta]` y su familia a `FAMILIA_MODELO`.
2. En `mibici/modeling/train.py`, escribir la función `entrenar_...(panel, tarea)`,
   que registra en el run activo, y añadirla a `ENTRENADORES`.
3. Si usa un hiperparámetro nuevo, declararlo en `MLproject` y en `train.main`
   con valor por defecto `config` (la prueba `test_mlproject` verifica que coincidan).

### Variables originales y tratamiento

La tabla limpia conserva únicamente `viaje_id`, `inicio_del_viaje`,
`fin_del_viaje`, `origen_id` y `destino_id`. Se normalizan nombres, fechas locales
y tipos de identificador; se eliminan registros inválidos y duplicados idénticos.
Un ID repetido con contenido contradictorio provoca un error.

La nomenclatura aporta `id`, `latitude`, `longitude`, `dpcapacity` y, cuando existe,
`name`. Se rechazan IDs duplicados y se identifican capacidades o coordenadas inválidas.

La ingesta usa bloques de 100,000 viajes, escritura Parquet y SQLite temporal para
deduplicar globalmente sin mantener todos los viajes crudos en memoria. La generación
de variables trabaja sobre el panel agregado. Es procesamiento local por bloques;
no se afirma ejecución distribuida ni uso de Spark.

## Partición temporal y feature engineering

| Partición | Fechas |
| --- | --- |
| Entrenamiento | 2025-10-01 a 2026-06-30 |
| Validación | 2026-07-01 a 2026-07-31 |
| Prueba | 2026-08-01 a 2026-09-30 |

No se barajan filas. Los rezagos se construyen antes de filtrar las particiones
para conservar el historial disponible al inicio de validación/prueba. La referencia
semanal y P2 requieren calentamiento de 168 horas. Para P2, cada ventana objetivo
debe terminar dentro de su partición: se excluyen sus últimas dos horas.

`features.py` contiene funciones reutilizables con docstrings:

- Panel horario completo con salidas, llegadas y flujo neto, incluidas horas en cero.
- Calendario y codificación cíclica; festivos opcionales solo con fuente documentada.
- Rezagos de 24/168 horas y medias móviles desplazadas hacia el pasado.
- Capacidad y flujo acumulado desde medianoche hasta antes de `t`, relativo a capacidad.
- Objetivo de P2 y cierre exclusivo de su ventana para filtrar sin invadir el siguiente corte.
- Perfil de 48 componentes horarios de P3, volumen diario y balance llegadas/salidas.

Las salidas, llegadas y flujo neto contemporáneos, así como las columnas `objetivo_`,
no son predictores de P1/P2. El acumulado diario mide desequilibrio desde medianoche,
no inventario ni rebalanceo nocturno. La anticipación admitida por el pipeline es
la predicción al inicio de `t`; la ventana futura de tres horas de P2 es un concepto
distinto. La escala y los perfiles de P3 se ajustan únicamente con entrenamiento.

## Pipeline, baselines y modelos candidatos

Flujo: **fuentes → ingesta/validación → Parquet + manifiesto → panel y variables →
partición temporal → evaluación → MLflow → resultados y evidencias**.

| Pregunta | Baseline ejecutado | Candidatas evaluadas | Métricas |
| --- | --- | --- | --- |
| P1 | Misma hora de la semana anterior; referencia adicional del día anterior. | Gradient boosting con variables históricas y validación temporal. | MAE en viajes/estación/hora y RMSE para complementar errores grandes. |
| P2 | Frecuencia de positivos por estación y hora en train; respaldo global de train para combinaciones nuevas. Umbral elegido en validación por máximo F1 y desempates por recall/precision. | Tres configuraciones de HistGradientBoostingClassifier. | Precision, recall, F1, average precision y prevalencia. |
| P3 | Perfiles elegibles de train, StandardScaler y K-Means; k entre 2 y 6 por silhouette. | PCA de ocho componentes + K-Means con k=3/4/6. | Silhouette, Davies-Bouldin, tamaños e interpretación de grupos. |

Las nueve candidatas y los cuatro baselines repetidos están documentados en
`reports/experiment_results.json`, dentro del subdirectorio del proyecto.
Las coordenadas de P3 se reservan para visualizar el mapa. La parte B incorpora
estabilidad entre semillas y periodos; la comparación y las limitaciones se
explican en el reporte de la segunda entrega. Las tablas siguientes conservan
los resultados de los baselines de la primera entrega.


### P1 · Evaluación por estación y hora

| Referencia | MAE validación | RMSE validación | MAE prueba | RMSE prueba |
| --- | ---: | ---: | ---: | ---: |
| Semanal, 168 h | 0.868982 | 1.824453 | 0.933745 | 1.921200 |
| Diaria, 24 h | 0.910708 | 1.966678 | 1.035465 | 2.194621 |

La referencia semanal tiene menor error en validación y prueba. Ambas referencias
se evalúan en las mismas 350,424 filas de validación y 689,544 de prueba. Las
gráficas de totales diarios son visualizaciones; no cambian la granularidad de las métricas.

### P2 · Umbral 0.30, seleccionado en validación

| Métrica | Validación | Prueba |
| --- | ---: | ---: |
| Precision | 0.403668 | 0.453065 |
| Recall | 0.314783 | 0.269737 |
| F1 | 0.353727 | 0.338152 |
| Average precision | 0.309132 | 0.325776 |
| Proporción de positivos | 0.016678 | 0.021816 |
| Filas evaluadas | 347,998 | 685,678 |

El baseline detecta aproximadamente el 27 % de las filas positivas del proxy en
prueba y omite alrededor del 73 %. Es una limitación para una futura priorización,
no una medición de estaciones realmente vacías/llenas. Las ventanas horarias se
solapan y no representan incidentes operativos independientes.

### P3 · Evaluación sobre perfiles de entrenamiento

| Medida | Resultado |
| --- | ---: |
| Grupos elegidos dentro del rango ensayado | 6 |
| Silhouette | 0.083076 |
| Davies-Bouldin | 1.953043 |
| Estaciones agrupadas | 424 |
| Estaciones excluidas por perfil no elegible | 47 |

El silhouette cercano a cero sugiere separación débil en el espacio de variables.
No se afirma una segmentación operativa consolidada ni un óptimo global de k.

## MLflow y evidencias

Capturas revisadas:

- [Lista de runs](tec-mlops-equipo-27/reports/figures/mlflow_01_runs.png).
- [Detalle: parámetros, métricas y etiquetas](tec-mlops-equipo-27/reports/figures/mlflow_02_detalle.png).
- [Artefacto registrado](tec-mlops-equipo-27/reports/figures/mlflow_03_artefacto.png).
- [Comparación de referencias de P1](tec-mlops-equipo-27/reports/figures/mlflow_04_comparacion.png).

## Referencias

Las referencias completas en formato APA 7 están en
[la propuesta de arquitectura](tec-mlops-equipo-27/references/arquitectura.md#referencias-apa-7).

- [MiBici: datos abiertos](https://mibici.net/es/datos-abiertos/).
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).
- [scikit-learn: evaluación de modelos](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [ML Canvas](https://www.ownml.co/).
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).
- [OpenAI](https://chatgpt.com)
- [Claude](https://claude.ai)
## Entrega 2 · Parte B: experimentación y artefactos

Desde la raíz Git, con el entorno fijo instalado y el servidor anterior activo:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo27
mlflow run . -e ingest --env-manager local
mlflow run . -e experiment --env-manager local
# Una sola pregunta (genera un resumen solo para esa pregunta):
mlflow run . -e experiment --env-manager local -P question=p1
# Una combinación individual:
mlflow run . -e train --env-manager local -P question=p1 -P model=hgb -P n_estimators=100 -P max_depth=6 -P learning_rate=0.1
mlflow run . -e train --env-manager local -P question=p3 -P model=pca_kmeans -P n_clusters=4 -P n_components=8
```

`--env-manager local` utiliza el entorno activado con las versiones fijas de
`requirements.txt`; `--env-manager uv` permite que MLflow cree el entorno declarado
en `python_env.yaml`. Ambos ejecutan los mismos puntos de entrada. Confirmar los
cambios de código en Git antes de entrenar: cada corrida exige un commit limpio.

`experiment` repite los cuatro baselines sin reemplazar el reporte histórico y
comprueba sus métricas contra `reports/baseline_results.json` con tolerancia
numérica de 1e-9. Reconstruye el panel desde los Parquet cuyos hashes verifica el
manifiesto de la primera entrega. No cambia la versión de datos ni los cortes.
Los nuevos IDs se vinculan a los IDs originales en `baseline_parity`.

En `mibici-equipo27`, la jerarquía es: corrida del Project, baseline por referencia,
una corrida `search-pN` por búsqueda y tres hijas candidatas por pregunta. Filtrar
por `question`, `model_family`, `stage_of_work`, `equipo`, `git_commit` y
`data_version`. P1/P2 prueban boosting de histogramas con 60/100/160 iteraciones,
profundidades 4/6/8 y tasas 0.1/0.1/0.08, sin early stopping aleatorio. P3 prueba
PCA (8 componentes) + K-Means (3/4/6 grupos). Todo se ajusta solo con train.

Cada baseline y candidata incluye un modelo cargable con firma y ejemplo numérico,
`config/` (configuración efectiva, fuentes de configuración y dependencias),
`data/data_manifest.json`, `plots/` y `results/`. `log_input` registra esquema y
digest de los agregados; no sube los viajes. P2 conserva el umbral que maximiza F1
de validación en el objeto desplegable; `predict` devuelve alertas 0/1 y
`predict_proba` permite priorizar por probabilidad.

P1 mantiene MAE/RMSE; P2 mantiene precisión, recall, F1 y AP y la purga de ventanas
que cruzan cortes. Para P3 se conserva silhouette/Davies-Bouldin en las 50 variables
escaladas con train, incluso al ajustar PCA. Se añaden validación/prueba temporal
sobre estaciones presentes en train, ARI entre periodos y ARI de tres semillas.
Estos diagnósticos amplían la primera entrega sin sustituir sus métricas de train.
La desviación de resultados semanales P1/P2 es descriptiva, sin asumir filas iid.
Se miden tiempo de ajuste, tamaño serializado y latencia por fila (mediana de tres
predicciones, lote de hasta 1,000 filas, después de calentamiento).

El resumen se guarda en `tec-mlops-equipo-27/reports/experiment_results.json` y como
artefacto de la corrida Project. Repetir `experiment` completo antes del registro
para que el resumen contenga las tres preguntas. Datos, base SQLite, artefactos
MLflow y estimadores serializados permanecen excluidos de Git.

## Entrega 2 · Parte C: selección y Model Registry

Después de `mlflow run . -e experiment` completo, ejecutar desde la raíz:

```bash
mlflow run . -e register --env-manager local
```

Se selecciona exclusivamente con validación: P1 minimiza MAE; P2 maximiza average
precision (priorización de alertas con clase rara); P3 maximiza silhouette temporal
en el espacio original escalado. Diferencias de hasta 1% relativo en P1/P2 o
0.005 puntos en P3 se consideran casi empate: P1 prefiere menor dispersión del
MAE semanal; P2 mayor recall, luego F1; P3 mayor ARI mínimo entre semillas, luego
ARI temporal. El tamaño del modelo desempata lo anterior. Se reportan también
costo de ajuste/latencia. La prueba no decide el modelo ni el umbral de P2.
No se vuelve a ajustar con train+validation: se registra exactamente el estimador
evaluado y se conserva el baseline cuando siga siendo la mejor alternativa.

Antes de registrar se verifica que hay tres candidatas y baseline por pregunta,
que los runs terminaron en el experimento correcto y que datos, commit, métricas,
cortes, modelos e inputs coinciden con el resumen. Las candidatas deben tener un
padre de búsqueda. Cambiar el JSON para inventar resultados provoca un error.

Usamos aliases, recomendados en el flujo vigente de MLflow, para que el consumo
no dependa del número de versión ni de las etapas clásicas obsoletas. Nombres:

- `models:/mibici-equipo27-p1-demanda@champion`
- `models:/mibici-equipo27-p2-riesgo@champion`
- `models:/mibici-equipo27-p3-grupos@champion`

Para reproducir los modelos elegidos de esta ejecución desde la raíz:

```bash
mlflow run . -e train --env-manager local -P question=p1 -P model=hgb -P n_estimators=160 -P max_depth=8 -P learning_rate=0.08
mlflow run . -e train --env-manager local -P question=p2 -P model=hgb -P n_estimators=160 -P max_depth=8 -P learning_rate=0.08
mlflow run . -e train --env-manager local -P question=p3 -P model=pca_kmeans -P n_clusters=4 -P n_components=8
```

Cada versión conserva descripción, experimento, corrida origen, commit del
entrenamiento, commit del registro y versión SHA-256 de datos. Primero se carga
por versión y se compara su predicción con el modelo origen; después se mueve
`champion` y se vuelve a cargar por alias. Si falla esta última comprobación, se
restaura el alias anterior. Una repetición reutiliza la versión de la misma corrida.
`champion` identifica el modelo de producción académica local; esta entrega no
levanta un servicio HTTP de predicción ni demuestra impacto en operación real.

Ejemplo de consumo, con un DataFrame que cumpla la firma registrada:

```python
import mlflow
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.pyfunc.load_model("models:/mibici-equipo27-p1-demanda@champion")
predictions = model.predict(X)
# P2: sklearn permite predict_proba; predict conserva el umbral de validación.
risk = mlflow.sklearn.load_model("models:/mibici-equipo27-p2-riesgo@champion")
probabilities = risk.predict_proba(X_risk)[:, 1]
# P3: perfiles de estaciones con las columnas de la firma del modelo.
groups = mlflow.pyfunc.load_model("models:/mibici-equipo27-p3-grupos@champion")
labels = groups.predict(X_profiles)
```

`X`, `X_risk` y `X_profiles` representan DataFrames preparados previamente con
las columnas y tipos de la firma de cada modelo; no son viajes crudos ni son
intercambiables entre preguntas. La firma y el ejemplo de entrada se pueden
consultar en el modelo registrado en la UI de MLflow. Ejecutar el ejemplo con
el entorno del proyecto y `MLFLOW_TRACKING_URI` definido para ese servidor.

El resultado verificable queda en `reports/registry_results.json`. Las bases de
tracking y modelos son locales; un profesor que clone el repositorio debe repetir
ingesta, experimentación y registro para reconstruirlos. Los IDs nuevos cambiarán;
se comparan métricas, hashes, parámetros y comportamiento, no IDs locales.

### Flujo de integración de esta entrega

Estado revisado el 25 de octubre de 2026:

1. Parte A: integrada en `main` mediante el PR #10.
2. Parte C: integrada en `feat/mlflow-experiments` mediante el PR #12.
3. Partes B y C: pendientes de integración conjunta desde
   `feat/mlflow-experiments` hacia `main`.

Las mejoras documentales de `docs/mlflow-registry` parten de
`feat/mlflow-experiments` y deben revisarse contra esa base mientras no esté
integrada en `main`. Después de integrar B y C, actualizar la base del PR de
documentación a `main` y comprobar su diff. No fusionar los PR sin revisión.
No se añaden commits ni revisiones a nombre de otros
integrantes: cada persona debe abrir y revisar al menos un PR desde su cuenta.
Los textos originales de PR y la validación local en
`tec-mlops-equipo-27/references/` conservan el contexto de su preparación;
sus notas de publicación pendiente no describen el estado actual de integración.
Las capturas y el reporte documentan la ejecución local original.

## Entrega 3 · Parte 1: Pruebas unitarias con PyTest

### Pruebas del proyecto

Esta carpeta contiene dos familias de pruebas que conviven y se ejecutan juntas:

* **PyTest** (`test_procesamiento_datos.py`, `test_feature_engineering.py`,
  `test_utilidades_entrenamiento.py` y `test_reproducibilidad.py`): pruebas nuevas,
  con casos normales y casos límite, y una prueba de reproducibilidad del entrenamiento.
* **unittest** (`test_dataset.py`, `test_features.py`, `test_baselines.py`,
  `test_experiments.py`, `test_registry.py` y `test_mlproject.py`): pruebas de
  contrato y de regresión del proyecto. PyTest también las recoge y ejecuta.

Todas utilizan datos sintéticos y archivos temporales: no requieren viajes
descargados, acceso a Internet ni un servidor de MLflow.

### Requisitos

Con el entorno del proyecto instalado (Python 3.13):

```bash
cd tec-mlops-equipo-27
python -m pip install -r requirements.txt
python -m pytest --version
```

`pytest` debe figurar en `requirements.txt` y, con la misma versión, en
`python_env.yaml` de la raíz del repositorio: `test_mlproject.py` verifica que ambos
archivos coincidan.

### Cómo ejecutar las pruebas

Siempre desde `tec-mlops-equipo-27/` (la carpeta que contiene `tests/`):

```bash
python -m pytest                 # un solo comando: toda la suite
python -m pytest -v              # una línea por prueba
```

Variantes útiles:

```bash
python -m pytest tests/test_procesamiento_datos.py -v        # un archivo
python -m pytest tests/test_feature_engineering.py -k rezago # por nombre
python -m pytest -x --tb=short                               # se detiene en el primer fallo
python -m pytest --collect-only -q                           # lista las pruebas sin ejecutarlas
python -m unittest discover -s tests -v                      # solo las pruebas unittest (50)
```

`make test` ejecuta únicamente la suite `unittest`. Resultado esperado de
`python -m pytest`: **231 pruebas aprobadas**, de las cuales 50 son `unittest` y 181
son de PyTest. En total tarda alrededor de 20 segundos. Las advertencias
(`warnings`) que pueden aparecer proceden de MLflow y pandas dentro de
`test_reproducibilidad.py`; no son fallos.

### Qué cubre cada archivo de PyTest

#### `test_procesamiento_datos.py` (60 pruebas): `mibici/dataset.py`

| Función | Casos normales | Casos límite |
|---|---|---|
| `nombre_columna` | Minúsculas, acentos y espacios. | Nombre vacío, valor que no es texto, separadores repetidos. |
| `normalizar_viajes` | Tipos, esquema final y exclusión de campos personales. | Inicio inclusivo y cierre exclusivo del periodo, IDs en 0, negativos, decimales, nulos o texto, fechas ilegibles, fin anterior al inicio, duración cero, tabla vacía, columnas faltantes o duplicadas tras normalizar, fechas con zona horaria. |
| `normalizar_estaciones` | Orden por `id`, nombre opcional. | Capacidad en 0, negativa, infinita, texto o nula; coordenadas fuera de rango y justo en el límite; IDs inválidos o duplicados; columna requerida ausente. |
| `version_manifest` | Hash SHA-256 determinista. | Ignora `data_version` y el orden de llaves; cambia con cualquier cambio de contenido. |
| `sha256_archivo` | Valores conocidos. | Archivo vacío y archivo mayor que un bloque de lectura. |
| `detectar_codificacion` | UTF-8 con BOM y cp1252. | Archivo vacío y bytes que solo `latin-1` acepta. |

#### `test_feature_engineering.py` (59 pruebas): `mibici/features.py`

| Función | Casos normales | Casos límite |
|---|---|---|
| `agregar_variables_calendario` | Hora, día, mes y codificación seno/coseno. | Fin de semana en viernes/sábado/domingo, contigüidad de las 23 h y las 0 h, festivos con hora, lista de festivos vacía, panel de entrada intacto. |
| `agregar_rezagos` | Desplazamiento de `k` horas, aislamiento por estación, independencia del orden de filas. | Rezago igual y menor al horizonte (fuga de información), rezagos vacíos, en 0, negativos, decimales o booleanos, huecos horarios, claves duplicadas, periodo sin tipo fecha, columna clave ausente. |
| `agregar_promedios_moviles` | Media de la ventana anterior al periodo, sin cruzar estaciones. | El periodo actual nunca entra en la media, ventana mayor que la serie, ventanas inválidas. |
| `construir_objetivo_riesgo` | Flujo futuro, etiquetas de ambas direcciones y cierre de ventana calculados a mano. | Umbral inclusivo, cola incompleta en NA, capacidad en 0, negativa, nula o infinita, estaciones sin mezclar, umbral inválido y umbral igual a 1, horizonte inválido, columnas ausentes. |
| `filtrar_periodo` | Fechas extremas inclusivas. | Sin límites, límites abiertos, rango invertido, resultado vacío, purga de ventanas que cruzan el corte, ventanas incompletas, `col_fin_objetivo` sin `fin`, columna inexistente. |

#### `test_utilidades_entrenamiento.py` (56 pruebas): `mibici/modeling/`

| Función o clase | Casos normales | Casos límite |
|---|---|---|
| `convertir_valor` | `int`, `float`, notación científica, texto. | `none`/`null`, espacios, cadena vacía, entrada que no es texto. |
| `resolver_hiperparametros` | Conversión de tipos, valor `config`. | La configuración global no se modifica, pregunta o modelo inexistente, parámetro que no aplica, modelo sin hiperparámetros. |
| `planear_tareas`, `resolver_etapa` | Normalización de mayúsculas y espacios, etapa por defecto. | `all` con modelo o parámetros no permitidos, `Tarea` inmutable, lista de tareas vacía, etapa inválida. |
| `metricas_clasificacion` | Valores calculados a mano, separación perfecta. | Umbral inclusivo, sin predicciones positivas, entrada booleana, una clase, tres clases o ninguna. |
| `frecuencias_riesgo` | Tasa por estación y hora. | Estación u hora no vistas: recurre al promedio global. |
| `elegir_umbral` | Desempate hacia el umbral mayor. | Mejor umbral en el extremo inferior, una sola clase. |
| `cluster_scores` | Grupos bien separados. | Un solo grupo, un grupo por punto, una sola muestra: la partición colapsada se marca. |
| `LagRegressor`, `FrequencyClassifier`, `ThresholdClassifier` | Predicciones y probabilidades. | Rezago con NA, columna ausente, estación u hora no vista, umbral en 0, en 1 y por encima de 1. |

#### `test_reproducibilidad.py` (6 pruebas): reproducibilidad del entrenamiento

Entrena **dos veces** el código real de `mibici.modeling.candidates` con la misma
semilla y comprueba que las métricas de calidad y las predicciones de los dos modelos
sean idénticas. Cubre P1 y P2 (HistGradientBoosting) y P3 (PCA + K-Means), cada una con
las semillas 42 y 7. Excluye únicamente las medidas de costo: tiempo de ajuste, latencia
y tamaño del modelo. Usa un panel sintético y una base SQLite temporal de MLflow, así
que no escribe en el repositorio.

```bash
python -m pytest tests/test_reproducibilidad.py -v
```

### Verificación del módulo de variables (unittest)

Las pruebas `unittest` incluyen una ejecución real del comando
`python -m mibici.features` con entradas y salida Parquet. Se verifican los conteos del
panel, horas sin viajes, estaciones sin nomenclatura, capacidades inválidas, uniones sin
multiplicación de registros, independencia entre estaciones, rezagos, ventanas móviles,
acumulado diario y límites temporales.

### Uso del corte temporal de P2

Construir primero las variables sobre el historial completo, para conservar el
contexto anterior a validación y prueba. Después filtrar cada partición:

```python
from mibici.config import FIN_ENTRENAMIENTO
from mibici.features import filtrar_periodo

train_p2 = filtrar_periodo(
    panel,
    fin=FIN_ENTRENAMIENTO,
    col_fin_objetivo="objetivo_fin_periodo",
)
```

Esto excluye las ventanas futuras que cruzan el corte y las que están incompletas.
Repetir para validación y prueba con sus fechas correspondientes. P1 puede filtrar
sin `col_fin_objetivo`, pues su objetivo solo cubre la hora de cada fila.

Al entrenar, seleccionar explícitamente los predictores: los conteos `salidas`,
`llegadas`, `flujo_neto` de la hora actual y las columnas `objetivo_` no deben entrar
en las variables de entrada de P1 ni de P2. Las filas sin capacidad válida carecen
de etiqueta de P2 y deberán reportarse y excluirse de su evaluación.

### Resultados de las pruebas:
````bash
(venv) tec-mlops-equipo-27 % python -m pytest
platform darwin -- Python 3.13.15, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/oscatorr/Documents/TEC/4to trimestre/ML Ops/tec-mlops-equipo-27/tec-mlops-equipo-27
configfile: pyproject.toml
plugins: platformdirs-4.13.3, anyio-4.15.1
collected 231 items                                                                                                                                                           

tests/test_baselines.py ...                                                                                                                                             [  1%]
tests/test_dataset.py ............                                                                                                                                      [  6%]
tests/test_experiments.py .....                                                                                                                                         [  8%]
tests/test_feature_engineering.py ...........................................................                                                                           [ 34%]
tests/test_features.py ................                                                                                                                                 [ 41%]
tests/test_mlproject.py .........                                                                                                                                       [ 45%]
tests/test_procesamiento_datos.py ............................................................                                                                          [ 70%]
tests/test_registry.py .....                                                                                                                                            [ 73%]
tests/test_reproducibilidad.py ......                                                                                                                                   [ 75%]
tests/test_utilidades_entrenamiento.py ........................................................                                                                         [100%]

tests/test_reproducibilidad.py::test_misma_semilla_reproduce_metricas_y_predicciones[P1-hgb-42]
  .../tec-mlops-equipo-27/tec-mlops-equipo-27/venv/lib/python3.13/site-packages/mlflow/store/tracking/sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

tests/test_reproducibilidad.py: 36 warnings
  .../tec-mlops-equipo-27/tec-mlops-equipo-27/venv/lib/python3.13/site-packages/mlflow/data/dataset_source_registry.py:148: UserWarning: The specified dataset source can be interpreted in multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource. MLflow will assume that this is a LocalArtifactDatasetSource source.
    return _dataset_source_registry.resolve(

tests/test_reproducibilidad.py: 36 warnings
  .../tec-mlops-equipo-27/tec-mlops-equipo-27/venv/lib/python3.13/site-packages/mlflow/data/digest_utils.py:29: Pandas4Warning:Starting with pandas version 4.0 all arguments of all will be keyword-only.
    string_columns = trimmed_df.columns[(df.map(type) == str).all(0)]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

### Chequeos básicos de seguridad

Desde la carpeta `tec-mlops-equipo-27`, con el entorno virtual activo:

```bash
python -m pip install -r requirements-security.txt
make security
```

El comando ejecuta Bandit y detect-secrets. La detección de secretos revisa archivos versionados y nuevos archivos pendientes, sin enviar posibles credenciales a servicios externos.

La baseline contiene únicamente hashes e identificadores de MLflow revisados como falsos positivos. Las excepciones de Bandit están limitadas y justificadas en las llamadas internas a Git.

Los mismos chequeos se ejecutan automáticamente en cada push y pull request.
