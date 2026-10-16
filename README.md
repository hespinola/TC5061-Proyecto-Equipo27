# Proyecto MiBici

Proyecto colaborativo de **TC5061** para diseñar una solución de Machine Learning aplicada al sistema de bicicleta pública MiBici del Área Metropolitana de Guadalajara.

**Equipo:** 27

**Integrantes:**
- Humberto Alejandro Espinola González (A01840066)
- Daniela Montserrat Enriquez Ballesteros (A01797865)

> **Estado inicial:** este README es el punto de partida para organizar al equipo. La estructura, la ingesta, el módulo de variables y los experimentos están pendientes de implementación. Las decisiones técnicas propuestas aquí se validarán entre los cuatro integrantes.

## Propósito

Construir una base reproducible para explorar y preparar millones de registros de viajes, proponer modelos de regresión, clasificación y clustering, y registrar una línea base por pregunta en MLflow. Esta primera entrega será el fundamento del proyecto integrador y de las siguientes entregas del curso.

El proyecto debe permitir reconstruir los datos utilizados, rastrear las decisiones del equipo y comparar los resultados de nuevos modelos con los experimentos base.

## Bienvenidos al equipo

Para empezar a colaborar:

1. Completar su nombre, usuario y responsabilidad en la tabla siguiente.
2. Revisar la actividad y acordar las preguntas, el periodo de datos y las herramientas.
3. Tomar una tarea de la lista de pendientes y crear un Issue con su alcance y criterio de aceptación.
4. Trabajar en una rama propia, abrir un Pull Request y revisar el trabajo de otro integrante.

Los roles propuestos distribuyen la coordinación; todos participamos en las decisiones, el código y las revisiones.

| Integrante | Usuario de GitHub | Responsabilidad propuesta |
| --- | --- | --- |
| Daniela Montserrat Enriquez Ballesteros | danimoneb | dependencias y módulo de *feature engineering* (`features.py`) |
| Pendiente | Pendiente | Pendiente |
| Pendiente | Pendiente | Pendiente |
| Pendiente | Pendiente | Pendiente |

## Datos y reproducibilidad

Fuente principal: [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), con archivos CSV mensuales de viajes y nomenclatura de estaciones. La actividad señala la [nomenclatura de septiembre de 2026](https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv) como referencia; su elección y compatibilidad con el periodo de viajes deben justificarse.

Antes de entrenar, registraremos:

- Periodo seleccionado, enlaces exactos de descarga y fecha de obtención.
- Tamaño real: meses descargados, viajes originales y válidos, y estaciones utilizadas.
- Columnas originales necesarias, reglas de limpieza y construcción de cada objetivo.
- Condiciones de uso de la fuente y tratamiento de los campos personales. Utilizaremos únicamente las variables necesarias y presentaremos resultados agregados.
- Fechas de entrenamiento, validación y prueba, con justificación de la partición.

**Propuesta de ingesta:** descargar desde `dataset.py`, normalizar columnas, codificaciones y fechas, validar los registros y generar archivos Parquet. El proceso deberá ejecutarse desde un clon limpio sin descargas o ediciones manuales.

**Propuesta de versionado:** guardar en Git un manifiesto en `references/data_manifest.json` con URLs, hashes SHA-256, conteos, periodo y versión del proceso. Registrar su identificador en MLflow. Si un archivo de origen cambia, la ingesta deberá detectar la discrepancia; para reconstruir la misma versión, el equipo deberá conservar los archivos originales en un almacenamiento compartido fuera de Git.

Las variables de predicción usarán solo información disponible en ese momento. Los rezagos se calcularán con datos anteriores y los transformadores se ajustarán con entrenamiento. Si usamos una nomenclatura reciente para viajes históricos, documentaremos la limitación de su capacidad y cobertura.

## Estructura prevista

Generaremos la base con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) y `include_code_scaffold = Yes`. El nombre de módulo propuesto es `mibici`; esta estructura todavía no está creada.

```text
mlops-equipo-27/
├── README.md
├── .gitignore
├── requirements.txt          # Propuesta: dependencias con versiones fijas
├── Makefile                  # Comandos comunes
├── data/                     # Datos locales, excluidos de Git
│   ├── raw/
│   ├── external/
│   ├── interim/
│   └── processed/
├── mibici/                   # Código reutilizable
│   ├── config.py             # Rutas, fechas, semillas y configuración
│   ├── dataset.py            # Descarga, validación y versionado
│   ├── features.py           # Transformaciones documentadas por pregunta
│   └── modeling/             # Entrenamiento y evaluación
├── notebooks/                # Exploración
├── models/                   # Modelos locales, excluidos de Git
├── references/               # ML Canvas, arquitectura y manifiesto
└── reports/
    └── figures/              # Figuras y evidencias de la propuesta
```

`features.py` contendrá funciones con docstrings que expliquen cada transformación, la pregunta que atiende y cómo evita utilizar información futura.

## Instalación y ejecución

### Preparación del proyecto

>Versión de Python utilizada para instalación inicial: 3.13

```bash
git clone https://github.com/hespinola/tec-mlops-equipo-27
cd tec-mlops-equipo-27
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Comandos del proyecto pendientes de implementar

Se proponen exponer los siguientes comandos y actualizar esta sección cuando se hayan probado desde un clon limpio:

| Acción | Comando previsto | Resultado esperado |
| --- | --- | --- |
| Ingesta reproducible | `make data` | Descarga, normalización, Parquet y manifiesto |
| Experimento base | `make baseline` | Al menos un run para P1, P2 y P3 |
| Consultar resultados locales | `make tracking` | Interfaz de MLflow |

**Estos comandos aún no están implementados ni verificados.** El periodo, las rutas, la semilla y la configuración de tracking se centralizarán en `mibici/config.py`. El README final incluirá los comandos reales y sus requisitos.

## Pipeline y experimentos

Flujo propuesto: **fuentes → ingesta y validación → datos versionados → limpieza y agregación → variables → entrenamiento → evaluación → MLflow**.

Para cada pregunta acordaremos un baseline sencillo, al menos un modelo candidato y métricas coherentes con la tarea y la decisión operativa. Los resultados están pendientes; no se han ejecutado experimentos.

En [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) usaremos un experimento con nombre `mibici-equipo27` y al menos un run por pregunta ejecutado desde el código del repositorio. Cada run registrará:

- Parámetros: periodo, versión de datos, partición, semilla y configuración del modelo.
- Métricas de evaluación acordadas para la pregunta.
- Etiquetas: equipo, pregunta y commit de Git.
- Al menos un artefacto, como una gráfica, tabla de evaluación o mapa de grupos.

Guardaremos evidencias de la lista de runs, el detalle de un run, un artefacto y la comparación de dos o más runs para el PDF de entrega.

## Colaboración con Git y GitHub

Seguiremos [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): rama de trabajo → commits → Pull Request → revisión → integración.

- **Rama principal:** `main`. Pendiente configurar su protección y exigir al menos una aprobación antes de integrar cambios.
- **Ramas:** `feat/descripcion`, `fix/descripcion` o `docs/descripcion`; por ejemplo, `feat/ingesta-mensual`.
- **Commits:** prefijos `feat:`, `fix:`, `docs:`, `refactor:` o `chore:`; por ejemplo, `docs: documentar fuentes de datos`.
- **Pull Requests:** indicar objetivo, cambios, cómo se verificaron y el Issue relacionado. Solicitar revisión de otro integrante y resolver sus comentarios.
- **Participación:** cada integrante deberá tener commits propios, abrir al menos un Pull Request y revisar al menos uno de otra persona.

Ejemplo para iniciar una contribución después de clonar el repositorio:

```bash
git checkout main
git pull --rebase origin main
git checkout -b docs/completar-integrantes
```

## Archivos que no se subirán a Git

Antes de agregar datos o ejecutar experimentos, ampliaremos `.gitignore` para excluir datos de MiBici, contenido de `models/`, `mlruns/`, `mlartifacts/`, bases locales de tracking, `.env`, credenciales y entornos virtuales. Se conservarán únicamente los marcadores de carpetas que requiera la plantilla.

Versionaremos código, configuración sin secretos, dependencias, manifiestos, documentación y evidencias seleccionadas. No basta con excluir un archivo después: los datos, modelos y credenciales tampoco deben quedar en el historial.

## Pendientes para la primera entrega

- [ ] Confirmar número de equipo, integrantes y acceso del profesor al repositorio.
- [ ] Generar la estructura con `ccds`, fijar dependencias y ampliar `.gitignore`.
- [ ] Proteger `main` y acordar responsabilidades e Issues.
- [ ] Incorporar las preguntas oficiales y completar el ML Canvas.
- [ ] Justificar periodo, objetivos, particiones y tratamiento de datos personales.
- [ ] Implementar y verificar la ingesta reproducible y el versionado.
- [ ] Crear al menos un notebook de exploración y el módulo `features.py`.
- [ ] Definir pipeline, baseline, modelos candidatos y métricas por pregunta.
- [ ] Guardar el ML Canvas y el diagrama de arquitectura justificado en `references/`.
- [ ] Ejecutar los tres baselines y registrar parámetros, métricas, etiquetas y artefactos.
- [ ] Capturar las cuatro evidencias de MLflow.
- [ ] Verificar instalación, ingesta y experimento base desde un clon limpio.
- [ ] Confirmar commits, un Pull Request abierto y una revisión por integrante.
- [ ] Completar este README con decisiones y comandos verificados.
- [ ] Preparar `TC5061_Propuesta_Proyecto_Equipo27.pdf` con el enlace al repositorio, propuesta, resultados, evidencias y referencias APA 7.

## Referencias de trabajo

- [MiBici: datos abiertos](https://mibici.net/es/datos-abiertos/).
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).
- [ML Canvas](https://www.ml-canvas.com/).
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).
- [scikit-learn: evaluación de modelos](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).
