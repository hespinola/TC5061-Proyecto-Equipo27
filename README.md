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

Las ramas y los textos de los PR están preparados localmente; su publicación,
revisión e integración siguen pendientes. Los tres modelos `champion` se
verificaron en el MLflow local de esta tarea.

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
Python y `make` de este documento se ejecutan desde ese subdirectorio.

```text
tec-mlops-equipo-27/                 # Raíz del repositorio Git
├── README.md                       # Este documento
├── MLproject                       # MLflow Project: puntos de entrada ingest y train
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
    │       └── train.py            # Baselines, evaluación y MLflow
    ├── notebooks/                  # Exploración
    ├── references/                 # Manifiesto, arquitectura y guía de evidencias
    ├── reports/
    │   ├── baseline_results.json   # Métricas e IDs de runs
    │   ├── resultados_base.md      # Interpretación
    │   └── figures/                # Gráficas y capturas seleccionadas
    ├── models/                    # Modelos locales, excluidos de Git
    └── tests/                     # Pruebas sin descargas de MiBici
```

La plantilla también conserva archivos auxiliares y carpetas para documentación.
`predict.py` y `plots.py` no forman parte del pipeline validado de esta entrega.

## Instalación

Requisitos: **Python 3.13**, Git, acceso HTTPS a MiBici y un entorno virtual.
Los ejemplos usan macOS/Linux y `make`. Se verificó `pip check` sin conflictos
en el entorno de ejecución del equipo; las dependencias principales tienen
versiones fijas en `requirements.txt`.

```bash
git clone https://github.com/hespinola/TC5061-Proyecto-Equipo27.git
cd TC5061-Proyecto-Equipo27/tec-mlops-equipo-27

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

La configuración local de tracking se puede personalizar mediante `.env`, siguiendo
`.env.example`. Por defecto, `MLFLOW_TRACKING_URI` apunta al servidor local
`http://127.0.0.1:5000` (ver [MLflow Project](#mlflow-project-y-servidor-de-tracking)).

## Ejecución reproducible

| Acción | Comando | Equivalente Python |
| --- | --- | --- |
| Pruebas locales | `make test` | `python -m unittest discover -s tests -v` |
| Ingesta y verificación de datos | `make data` | `python -m mibici.dataset` |
| Panel y variables | `make features` | `python -m mibici.preparar` |
| Baselines (run padre + 4 nested runs) | `make baseline` | `python -m mibici.modeling.train` |
| Servidor de tracking y UI | `make tracking` | `python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` |

Orden de ejecución:

```bash
make test
make data
make data
make features
make baseline
make tracking
```

La segunda ingesta debe reproducir el manifiesto fijado. `make baseline` verifica
los Parquet y reconstruye el panel antes de evaluar, por lo que `make features` es
una etapa de inspección útil pero no necesita repetirse manualmente para entrenar.
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
terminal aparte, desde `tec-mlops-equipo-27/`:

```bash
make tracking
# equivale a:
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

La UI queda en [http://127.0.0.1:5000](http://127.0.0.1:5000). `mlflow.db`,
`mlruns/` y `mlartifacts/` están en `.gitignore` y no se versionan.

En la terminal donde se ejecutan las corridas:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo27
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

Desde la raíz del repositorio y con el servidor activo:

```bash
# Ingesta: descarga, verifica el manifiesto y lo registra como artefacto data/.
mlflow run . -e ingest --env-manager uv

# Los cuatro baselines (P1 168 h y 24 h, P2, P3) como nested runs de un run padre.
mlflow run . -e train --env-manager uv -P question=all

# Una pregunta y un modelo; los hiperparámetros no indicados salen de config.py.
mlflow run . -e train --env-manager uv -P question=p1 -P model=baseline -P rezago_horas=24
mlflow run . -e train --env-manager uv -P question=p3 -P n_clusters=4
```

El entrenamiento exige el código confirmado en Git (incluidos `MLproject` y
`python_env.yaml`). Los comandos del modelo elegido de cada pregunta se agregan
aquí al cerrar la experimentación.

### 4. Parámetros del punto de entrada `train`

| Parámetro | Valor por defecto | Uso |
| --- | --- | --- |
| `question` | `all` | `p1`, `p2`, `p3` o `all` (los cuatro baselines). |
| `model` | `config` → `baseline` | Modelo registrado en `config.HIPERPARAMETROS`. |
| `stage_of_work` | `config` → `baseline` o `candidate` | Etiqueta para filtrar: `baseline`, `search`, `candidate`. |
| `rezago_horas` | `config` → 168 | P1 baseline: 24 o 168. |
| `n_clusters` | `config` → `None` | P3: k fijo; `none` elige k por silhouette. |
| `n_estimators`, `max_depth`, `learning_rate` | `config` | Reservados para los modelos candidatos. |

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

| Pregunta | Baseline ejecutado | Candidato propuesto | Métricas |
| --- | --- | --- | --- |
| P1 | Misma hora de la semana anterior; referencia adicional del día anterior. | Gradient boosting con variables históricas y validación temporal. | MAE en viajes/estación/hora y RMSE para complementar errores grandes. |
| P2 | Frecuencia de positivos por estación y hora en train; respaldo global de train para combinaciones nuevas. Umbral elegido en validación por máximo F1 y desempates por recall/precision. | Regresión logística o gradient boosting. | Precision, recall, F1, average precision y prevalencia. |
| P3 | Perfiles elegibles de train, StandardScaler y K-Means; k entre 2 y 6 por silhouette. | Clustering jerárquico sobre las mismas variables y escala. | Silhouette, Davies-Bouldin, tamaños e interpretación de grupos. |

Los modelos candidatos son parte de la propuesta; **no se presentan como entrenados**.
Las coordenadas de P3 se reservan para visualizar el mapa. La elección de k en
train no equivale a medir desempeño supervisado en test; el estudio de estabilidad
entre semillas/periodos queda para una entrega posterior.


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
model = mlflow.pyfunc.load_model("models:/mibici-equipo27-p1-demanda@champion")
predictions = model.predict(X)
# P2: sklearn permite predict_proba; predict conserva el umbral de validación.
risk = mlflow.sklearn.load_model("models:/mibici-equipo27-p2-riesgo@champion")
probabilities = risk.predict_proba(X_risk)[:, 1]
```

El resultado verificable queda en `reports/registry_results.json`. Las bases de
tracking y modelos son locales; un profesor que clone el repositorio debe repetir
ingesta, experimentación y registro para reconstruirlos. Los IDs nuevos cambiarán;
se comparan métricas, hashes, parámetros y comportamiento, no IDs locales.

### Flujo de integración de esta entrega

1. PR de Dani: `feat/mlflow-project` hacia `main` (parte A).
2. PR B: `feat/mlflow-experiments` hacia `feat/mlflow-project`.
3. PR C: `feat/mlflow-registry` hacia `feat/mlflow-experiments`.

Después de integrar la base de cada PR se actualiza su destino a `main`, sin
fusionar los PR sin revisión. No se añaden commits ni revisiones a nombre de otros
integrantes: cada persona debe abrir y revisar al menos un PR desde su cuenta.
Consultar los textos preparados en `references/PR-parte-B.md` y
`references/PR-parte-C.md`; las capturas y el reporte documentan evidencia local.
