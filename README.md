# Proyecto MiBici · Equipo 27

**TC5061 · Primera entrega del proyecto integrador**

Propuesta y base reproducible de Machine Learning para analizar la demanda, el
desequilibrio de flujo y los patrones de uso de estaciones de MiBici, el sistema
de bicicleta pública del Área Metropolitana de Guadalajara.

La implementación procesó **4,497,119 viajes válidos**, construyó un panel de
**4,125,960 filas estación × hora** y registró cuatro runs de referencia en MLflow.
La ingesta se ejecutó dos veces con la misma versión de datos.

La estructura se generó con Cookiecutter Data Science y su módulo se llama `mibici`.

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

Mientras la implementación está en revisión, clonar su rama publicada:

```bash
git clone https://github.com/hespinola/tec-mlops-equipo-27.git
cd tec-mlops-equipo-27/tec-mlops-equipo-27

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

La configuración local de tracking se puede personalizar mediante `.env`, siguiendo
`.env.example`. Para reproducir el flujo local descrito, usar el valor por defecto
de `MLFLOW_TRACKING_URI`, sin configurar un servidor externo.

## Ejecución reproducible

| Acción | Comando | Equivalente Python |
| --- | --- | --- |
| Pruebas locales | `make test` | `python -m unittest discover -s tests -v` |
| Ingesta y verificación de datos | `make data` | `python -m mibici.dataset` |
| Panel y variables | `make features` | `python -m mibici.preparar` |
| Cuatro runs de baselines | `make baseline` | `python -m mibici.modeling.train` |
| UI local de MLflow | `make tracking` | `python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` |

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
