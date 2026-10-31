"""Entrenamiento P1/P2/P3 con selección en validación y trazabilidad en MLflow.

Es el punto de entrada ``train`` del MLflow Project (ver ``MLproject`` en la raíz
del repositorio)::

    mlflow run . -e train -P question=p1 -P model=baseline

Cada ejecución usa un run padre. Si la tarea produce un solo modelo, se registra
en ese run; si produce varios (``--question all``), cada uno es un run hijo
(nested run). Los valores ``config`` se sustituyen por los de ``mibici.config``.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import os
import shutil

# Git se ejecuta sin shell y con argumentos internos.
import subprocess  # nosec B404
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from mibici import features as f
from mibici.config import (
    DATA_MANIFEST_PATH,
    EQUIPO,
    ETAPAS_DE_TRABAJO,
    FAMILIA_MODELO,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    HIPERPARAMETROS,
    K_GRUPOS_CANDIDATOS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELO_POR_DEFECTO,
    MUESTRA_SILHOUETTE,
    PREGUNTAS,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
    REZAGOS_HORAS,
    SEED,
    UMBRAL_RIESGO,
    UMBRALES_CLASIFICACION,
    VALOR_DESDE_CONFIG,
    VENTANA_RIESGO_HORAS,
)
from mibici.preparar import preparar, verificar_datos

app = typer.Typer()

# Archivos que definen cómo se ejecuta una corrida; deben estar confirmados en Git.
ARCHIVOS_REPRODUCIBLES = (
    "mibici",
    "requirements.txt",
    "pyproject.toml",
    "../MLproject",
    "../python_env.yaml",
)


# ---------------------------------------------------------------------------
# Parámetros del MLflow Project
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tarea:
    """Un modelo a entrenar: pregunta (p1/p2/p3), nombre del modelo e hiperparámetros."""

    pregunta: str
    modelo: str
    hiperparametros: dict


def convertir_valor(texto: str):
    """Convierte el texto recibido por línea de comandos a None, int, float o str."""
    limpio = str(texto).strip()
    if limpio.lower() in {"none", "null"}:
        return None
    for tipo in (int, float):
        try:
            return tipo(limpio)
        except ValueError:
            pass
    return limpio


def resolver_hiperparametros(pregunta: str, modelo: str, valores: dict) -> dict:
    """Combina los valores por defecto de config.py con los recibidos en ``-P``.

    Un valor igual a ``VALOR_DESDE_CONFIG`` conserva el de config.py. Un
    hiperparámetro que no aplica al modelo es un error, para no registrar corridas
    con parámetros ignorados en silencio.
    """
    if pregunta not in HIPERPARAMETROS:
        raise ValueError(f"Pregunta '{pregunta}' no válida. Opciones: {', '.join(PREGUNTAS)}.")
    modelos = HIPERPARAMETROS[pregunta]
    if modelo not in modelos:
        raise ValueError(
            f"Modelo '{modelo}' no disponible para {pregunta}. Opciones: {', '.join(modelos)}."
        )
    resueltos = dict(modelos[modelo])
    for nombre, texto in valores.items():
        if texto == VALOR_DESDE_CONFIG:
            continue
        if nombre not in resueltos:
            admitidos = ", ".join(resueltos) or "ninguno"
            raise ValueError(
                f"'{nombre}' no aplica a {pregunta}/{modelo}. Hiperparámetros: {admitidos}."
            )
        resueltos[nombre] = convertir_valor(texto)
    return resueltos


def planear_tareas(question: str, model: str, valores: dict) -> list[Tarea]:
    """Traduce los parámetros del punto de entrada en la lista de modelos a entrenar."""
    question = question.strip().lower()
    model = MODELO_POR_DEFECTO if model == VALOR_DESDE_CONFIG else model.strip().lower()
    if question == "all":
        if model != MODELO_POR_DEFECTO or any(v != VALOR_DESDE_CONFIG for v in valores.values()):
            raise ValueError("--question all solo ejecuta los baselines con valores de config.py.")
        # Se conservan las dos referencias de P1 de la primera entrega (168 h y 24 h).
        tareas = [
            Tarea("p1", MODELO_POR_DEFECTO, {"rezago_horas": r})
            for r in sorted(REZAGOS_HORAS, reverse=True)
        ]
        return tareas + [
            Tarea(p, MODELO_POR_DEFECTO, resolver_hiperparametros(p, MODELO_POR_DEFECTO, {}))
            for p in ("p2", "p3")
        ]
    if question not in PREGUNTAS:
        raise ValueError(
            f"Pregunta '{question}' no válida. Opciones: {', '.join(PREGUNTAS)}, all."
        )
    return [Tarea(question, model, resolver_hiperparametros(question, model, valores))]


def resolver_etapa(stage_of_work: str, tareas: list[Tarea]) -> str:
    """Etiqueta ``stage_of_work``: baseline/candidate por defecto, o la indicada."""
    if stage_of_work == VALOR_DESDE_CONFIG:
        return "baseline" if all(t.modelo == "baseline" for t in tareas) else "candidate"
    if stage_of_work not in ETAPAS_DE_TRABAJO:
        raise ValueError(f"stage_of_work debe ser uno de: {', '.join(ETAPAS_DE_TRABAJO)}.")
    return stage_of_work


# ---------------------------------------------------------------------------
# Reproducibilidad y MLflow
# ---------------------------------------------------------------------------
def informacion_codigo():
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("Git debe estar instalado para registrar la versión del código.")

    def git(*args):
        # Los argumentos se definen abajo; no provienen de entradas del usuario.
        command = [git_executable, *args]
        output = subprocess.check_output(  # nosec B603
            command, cwd=PROJ_ROOT, text=True, shell=False
        )
        return output.strip()

    commit = git("rev-parse", "HEAD")
    # Un run debe identificar exactamente código confirmado; el manifiesto sí puede
    # ser recién generado y su hash se registra de forma independiente.
    cambios = git("status", "--porcelain", "--", *ARCHIVOS_REPRODUCIBLES)
    if cambios:
        raise ValueError("Confirma el código en Git antes de ejecutar corridas reproducibles.")
    return commit


def preparar_experimento():
    """Apunta al servidor de tracking y valida el experimento del run de ``mlflow run``."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id:
        client = mlflow.MlflowClient()
        experimento = client.get_experiment(client.get_run(run_id).info.experiment_id)
        if experimento.name != MLFLOW_EXPERIMENT_NAME:
            raise ValueError(
                f"mlflow run creó el run en el experimento '{experimento.name}'. Ejecuta con "
                f"--experiment-name {MLFLOW_EXPERIMENT_NAME} "
                f"(o export MLFLOW_EXPERIMENT_NAME={MLFLOW_EXPERIMENT_NAME})."
            )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def registrar_contexto(tarea: Tarea, panel, etapa: str):
    """Etiquetas, parámetros comunes y artefactos de configuración y datos de un run."""
    contexto = panel.attrs
    familia = FAMILIA_MODELO.get((tarea.pregunta, tarea.modelo), tarea.modelo)
    mlflow.set_tags(
        {
            "equipo": EQUIPO,
            "pregunta": tarea.pregunta.upper(),
            "question": tarea.pregunta,
            "model": tarea.modelo,
            "model_family": familia,
            "stage_of_work": etapa,
            "tipo": etapa,
            "git_commit": contexto["git_commit"],
            "data_version": contexto["data_version"],
            "datos": "MiBici_real",
            "evaluacion": "historica_offline",
        }
    )
    mlflow.log_params(
        {
            "data_version": contexto["data_version"],
            "seed": SEED,
            "periodo_inicio": contexto["inicio"],
            "fin_exclusivo": contexto["fin_exclusivo"],
            "fin_train": FIN_ENTRENAMIENTO,
            "fin_validation": FIN_VALIDACION,
            "ventana_riesgo_horas": VENTANA_RIESGO_HORAS,
            "umbral_riesgo": UMBRAL_RIESGO,
            "codigo_sha256": contexto["codigo_sha256"],
            # Valores efectivos; los parámetros de MLproject pueden decir "config".
            **{f"hp_{k}": v for k, v in tarea.hiperparametros.items()},
        }
    )
    for path in (
        PROJ_ROOT / "mibici/config.py",
        PROJ_ROOT / "requirements.txt",
        PROJ_ROOT.parent / "MLproject",
        PROJ_ROOT.parent / "python_env.yaml",
    ):
        mlflow.log_artifact(str(path), artifact_path="config")
    mlflow.log_dict(
        {p: version(p) for p in ("pandas", "numpy", "pyarrow", "scikit-learn", "mlflow")},
        "config/entorno.json",
    )
    mlflow.log_dict(
        {
            "pregunta": tarea.pregunta,
            "modelo": tarea.modelo,
            "model_family": familia,
            "stage_of_work": etapa,
            "hiperparametros": tarea.hiperparametros,
            "seed": SEED,
            "particion": {
                "inicio": contexto["inicio"],
                "fin_train": FIN_ENTRENAMIENTO,
                "fin_validation": FIN_VALIDACION,
                "fin_exclusivo": contexto["fin_exclusivo"],
            },
            "git_commit": contexto["git_commit"],
            "data_version": contexto["data_version"],
        },
        "config/run_config.json",
    )
    # Versión de los datos: el manifiesto de la ingesta (hashes y conteos, sin viajes).
    mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")


def registrar_dataset(df: pd.DataFrame, panel, split: str, objetivo: str | None = None):
    """Registra un conjunto como input de MLflow (esquema y digest, no las filas)."""
    dataset = mlflow.data.from_pandas(
        df,
        source=f"{DATA_MANIFEST_PATH.name}#{panel.attrs['data_version']}",
        name=f"mibici-panel-{split}-{panel.attrs['data_version'][:12]}",
        targets=objetivo,
    )
    mlflow.log_input(dataset, context=split)


def guardar_figura(nombre):
    """Registra la figura activa en ``plots/`` del run, sin escribir en reports/."""
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), f"plots/{nombre}")
    plt.close()


# ---------------------------------------------------------------------------
# Particiones y métricas
# ---------------------------------------------------------------------------
def particiones(panel, fin_exclusivo, p2=False):
    kwargs = {"col_fin_objetivo": "objetivo_fin_periodo"} if p2 else {}
    val_inicio = str((pd.Timestamp(FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)).date())
    test_inicio = str((pd.Timestamp(FIN_VALIDACION) + pd.Timedelta(days=1)).date())
    ultimo = str((pd.Timestamp(fin_exclusivo) - pd.Timedelta(days=1)).date())
    partes = {
        "train": f.filtrar_periodo(panel, fin=FIN_ENTRENAMIENTO, **kwargs),
        "validation": f.filtrar_periodo(panel, inicio=val_inicio, fin=FIN_VALIDACION, **kwargs),
        "test": f.filtrar_periodo(panel, inicio=test_inicio, fin=ultimo, **kwargs),
    }
    for nombre, df in partes.items():
        if df.empty:
            raise ValueError(f"Partición {nombre} vacía; revisar fechas y cobertura.")
    return partes


def metricas_clasificacion(y, prob, umbral):
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("P2 necesita ambas clases para evaluar average precision con sentido.")
    pred = prob >= umbral
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "average_precision": average_precision_score(y, prob),
        "positive_rate": float(y.mean()),
    }


def frecuencias_riesgo(train, datos):
    tasas = train.groupby(["estacion_id", "hora_del_dia"]).objetivo_en_riesgo.mean()
    claves = pd.MultiIndex.from_frame(datos[["estacion_id", "hora_del_dia"]])
    return (
        tasas.reindex(claves).fillna(float(train.objetivo_en_riesgo.mean())).to_numpy(dtype=float)
    )


# ---------------------------------------------------------------------------
# Modelos. Cada función registra en el run activo y devuelve un resumen.
# ---------------------------------------------------------------------------
def evaluar_p1(panel, tarea: Tarea):
    rezago = tarea.hiperparametros["rezago_horas"]
    if rezago not in REZAGOS_HORAS:
        raise ValueError(f"rezago_horas debe ser uno de {REZAGOS_HORAS} (ver config.py).")
    columna = f"salidas_rezago_{rezago}"
    partes = particiones(panel.dropna(subset=[columna]), panel.attrs["fin_exclusivo"])
    nombre = f"P1-baseline-rezago-{rezago}h"
    mlflow.set_tag("mlflow.runName", nombre)
    # El rezago queda como hp_rezago_horas (registrar_contexto).
    registrar_dataset(partes["train"], panel, "train", "salidas")
    metricas = {"train_filas": len(partes["train"])}
    for split in ("validation", "test"):
        df = partes[split]
        metricas.update(
            {
                f"{split}_mae": mean_absolute_error(df.salidas, df[columna]),
                f"{split}_rmse": root_mean_squared_error(df.salidas, df[columna]),
                f"{split}_filas": len(df),
            }
        )
    mlflow.log_metrics(metricas)
    diario = partes["test"].groupby(partes["test"].periodo.dt.date)[["salidas", columna]].sum()
    diario.plot(figsize=(9, 3), title=f"P1 · viajes diarios, prueba · rezago {rezago} h")
    guardar_figura(f"p1_rezago_{rezago}.png")
    return {"pregunta": "P1", "modelo": nombre, "metricas": metricas}


def evaluar_p2(panel, tarea: Tarea):
    elegibles = panel.dropna(subset=["objetivo_en_riesgo", "salidas_rezago_168"])
    partes = particiones(elegibles, panel.attrs["fin_exclusivo"], p2=True)
    train, val, test = (partes[x] for x in ("train", "validation", "test"))
    prob_val = frecuencias_riesgo(train, val)
    tabla = [
        {"umbral": u, **metricas_clasificacion(val.objetivo_en_riesgo, prob_val, u)}
        for u in UMBRALES_CLASIFICACION
    ]
    # Criterio reproducible: máximo F1; empate por recall, luego precision.
    elegido = max(tabla, key=lambda r: (r["f1"], r["recall"], r["precision"], r["umbral"]))
    mlflow.set_tag("mlflow.runName", "P2-baseline-frecuencia-estacion-hora")
    mlflow.log_params(
        {
            "umbral_clasificacion": elegido["umbral"],
            "criterio_umbral": "max validation_f1; desempate recall, precision",
            "fallback": "frecuencia_global_train",
        }
    )
    registrar_dataset(train, panel, "train", "objetivo_en_riesgo")
    metricas = {
        "train_filas": len(train),
        "train_positive_rate": float(train.objetivo_en_riesgo.mean()),
        "filas_sin_capacidad_valida": int(panel.capacidad.isna().sum()),
    }
    for split, df in (("validation", val), ("test", test)):
        prob = frecuencias_riesgo(train, df)
        metricas.update(
            {
                f"{split}_{k}": v
                for k, v in metricas_clasificacion(
                    df.objetivo_en_riesgo, prob, elegido["umbral"]
                ).items()
            }
        )
        metricas[f"{split}_filas"] = len(df)
    mlflow.log_metrics(metricas)
    mlflow.log_dict({"umbrales_validation": tabla}, "results/seleccion_umbral.json")
    prob = frecuencias_riesgo(train, test)
    matriz = confusion_matrix(test.objetivo_en_riesgo.astype(int), prob >= elegido["umbral"])
    plt.figure(figsize=(4, 3))
    plt.imshow(matriz, cmap="Blues")
    for (i, j), n in np.ndenumerate(matriz):
        plt.text(j, i, f"{n:,}", ha="center", va="center")
    plt.xticks([0, 1], ["Sin alerta", "Alerta"])
    plt.yticks([0, 1], ["Sin desequilibrio", "Desequilibrio"])
    plt.xlabel("Predicción")
    plt.ylabel("Proxy observado")
    plt.title("P2 · matriz de confusión, prueba")
    guardar_figura("p2_confusion.png")
    return {
        "pregunta": "P2",
        "modelo": "frecuencia-estacion-hora",
        "metricas": metricas,
        "umbral": elegido["umbral"],
    }


def evaluar_p3(panel, tarea: Tarea):
    n_clusters = tarea.hiperparametros.get("n_clusters")
    train = f.filtrar_periodo(panel, fin=FIN_ENTRENAMIENTO)
    nom = pd.read_parquet(PROCESSED_DATA_DIR / "nomenclatura.parquet")
    perfiles = f.construir_perfil_estaciones(train, nom)
    columnas = [c for c in perfiles if c.startswith("perfil_")] + [
        "salidas_promedio_dia",
        "balance_llegadas_salidas",
    ]
    # Estaciones sin salidas no tienen perfil interpretable y se reportan.
    validos = (perfiles.salidas_promedio_dia > 0) & perfiles[columnas].notna().all(axis=1)
    excluidas = int((~validos).sum())
    perfiles = perfiles.loc[validos].copy()
    X = StandardScaler().fit_transform(perfiles[columnas])
    # n_clusters=None: k por silhouette entre los candidatos de config.py.
    ks = K_GRUPOS_CANDIDATOS if n_clusters is None else (int(n_clusters),)
    candidatos = []
    modelos = {}
    for k in ks:
        if k >= len(X):
            continue
        modelo = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        if len(np.unique(modelo.labels_)) != k:
            continue
        s = silhouette_score(
            X, modelo.labels_, sample_size=min(len(X), MUESTRA_SILHOUETTE), random_state=SEED
        )
        candidatos.append({"k": k, "silhouette": float(s)})
        modelos[k] = modelo
    if not candidatos:
        raise ValueError("No hay estaciones/perfiles distintos suficientes para comparar grupos.")
    elegido = max(candidatos, key=lambda r: (r["silhouette"], -r["k"]))
    modelo = modelos[elegido["k"]]
    sufijo = "" if n_clusters is None else f"-k{elegido['k']}"
    mlflow.set_tag("mlflow.runName", f"P3-baseline-kmeans-perfiles-train{sufijo}")
    mlflow.log_params(
        {
            "k": elegido["k"],
            "n_init": 10,
            "escalado": "StandardScaler_train",
            "coordenadas_en_modelo": False,
            "seleccion_k": "max silhouette_train" if n_clusters is None else "fijo",
        }
    )
    registrar_dataset(perfiles[columnas], panel, "train")
    metricas = {
        "train_silhouette": elegido["silhouette"],
        "train_davies_bouldin": davies_bouldin_score(X, modelo.labels_),
        "estaciones_agrupadas": len(X),
        "estaciones_excluidas": excluidas,
    }
    mlflow.log_metrics(metricas)
    perfiles["grupo"] = modelo.labels_
    mlflow.log_dict(
        {
            str(g): {c: float(v) for c, v in fila.items()}
            for g, fila in perfiles.groupby("grupo")[columnas].mean().to_dict("index").items()
        },
        "results/perfiles_medios_grupos.json",
    )
    mlflow.log_dict(
        {
            "candidatos": candidatos,
            "variables": columnas,
            "tamano_grupos": {str(k): int(v) for k, v in perfiles.grupo.value_counts().items()},
        },
        "results/seleccion_grupos.json",
    )
    plt.figure(figsize=(7, 4))
    plt.scatter(perfiles.longitud, perfiles.latitud, c=perfiles.grupo, cmap="tab10", s=24)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("P3 · grupos de patrones de uso (coordenadas solo para visualizar)")
    guardar_figura("p3_grupos.png")
    return {"pregunta": "P3", "modelo": "kmeans", "metricas": metricas, "k": elegido["k"]}


# (pregunta, modelo) -> función. Los modelos nuevos se agregan aquí y en config.py.
from mibici.modeling.candidates import completar_baseline, evaluar_candidato

ENTRENADORES = {
    ("p1", "baseline"): evaluar_p1,
    ("p2", "baseline"): evaluar_p2,
    ("p3", "baseline"): evaluar_p3,
    ("p1", "hgb"): evaluar_candidato,
    ("p2", "hgb"): evaluar_candidato,
    ("p3", "pca_kmeans"): evaluar_candidato,
}


def ejecutar_tarea(tarea: Tarea, panel, etapa: str):
    entrenador = ENTRENADORES.get((tarea.pregunta, tarea.modelo))
    if entrenador is None:
        raise ValueError(f"{tarea.pregunta}/{tarea.modelo} está en config.py sin implementación.")
    registrar_contexto(tarea, panel, etapa)
    inicio_tarea = time.perf_counter()
    resultado = entrenador(panel, tarea)
    if tarea.modelo == "baseline":
        completar_baseline(panel, tarea, resultado)
    resultado["metricas"]["run_seconds"] = time.perf_counter() - inicio_tarea
    mlflow.log_metric("run_seconds", resultado["metricas"]["run_seconds"])
    resultado["run_id"] = mlflow.active_run().info.run_id
    mlflow.log_dict(resultado, "results/resumen.json")
    return resultado


def cargar_panel(commit):
    manifest = verificar_datos()
    # Reconstruir para no reutilizar un panel obsoleto respecto a código o datos.
    panel = preparar()
    h = hashlib.sha256()
    for path in sorted((PROJ_ROOT / "mibici").rglob("*.py")):
        h.update(str(path.relative_to(PROJ_ROOT)).encode())
        h.update(path.read_bytes())
    panel.attrs = {
        "git_commit": commit,
        "codigo_sha256": h.hexdigest(),
        "data_version": manifest["data_version"],
        "inicio": manifest["inicio"],
        "fin_exclusivo": manifest["fin_exclusivo"],
    }
    return panel


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all (los cuatro baselines)."),
    model: str = typer.Option(VALOR_DESDE_CONFIG, help="Modelo definido en config.py."),
    stage_of_work: str = typer.Option(VALOR_DESDE_CONFIG, help="baseline, search o candidate."),
    rezago_horas: str = typer.Option(VALOR_DESDE_CONFIG, help="P1 baseline: 24 o 168."),
    n_clusters: str = typer.Option(VALOR_DESDE_CONFIG, help="P3: k fijo; none = silhouette."),
    n_estimators: str = typer.Option(VALOR_DESDE_CONFIG, help="Modelos de ensamble."),
    max_depth: str = typer.Option(
        VALOR_DESDE_CONFIG, help="Profundidad máxima; none = sin límite."
    ),
    learning_rate: str = typer.Option(VALOR_DESDE_CONFIG, help="Gradient boosting."),
    n_components: str = typer.Option(VALOR_DESDE_CONFIG, help="Dimensiones PCA de P3."),
):
    """Entrena los modelos pedidos y los registra en el experimento del equipo."""
    valores = {
        "rezago_horas": rezago_horas,
        "n_clusters": n_clusters,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_components": n_components,
    }
    # Validar parámetros, Git y experimento antes del procesamiento costoso.
    tareas = planear_tareas(question, model, valores)
    etapa = resolver_etapa(stage_of_work, tareas)
    commit = informacion_codigo()
    preparar_experimento()
    inicio = time.perf_counter()
    panel = cargar_panel(commit)
    manifest = verificar_datos()
    # Con mlflow run, start_run() retoma el run creado por el proyecto (MLFLOW_RUN_ID).
    with mlflow.start_run() as padre:
        os.environ.pop("MLFLOW_RUN_ID", None)
        if len(tareas) == 1:
            resultados = [ejecutar_tarea(tareas[0], panel, etapa)]
        else:
            mlflow.set_tags(
                {
                    "mlflow.runName": "baselines-p1-p2-p3",
                    "equipo": EQUIPO,
                    "question": "all",
                    "stage_of_work": etapa,
                    "git_commit": commit,
                    "data_version": manifest["data_version"],
                }
            )
            resultados = []
            for tarea in tareas:
                with mlflow.start_run(nested=True):
                    resultados.append(ejecutar_tarea(tarea, panel, etapa))
        reporte = {
            "estado": "ejecutado",
            "fecha_utc": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "data_version": manifest["data_version"],
            "run_padre": padre.info.run_id,
            "segundos": time.perf_counter() - inicio,
            "runs": resultados,
        }
    if question.strip().lower() == "all":
        # Resultado de referencia versionado en Git desde la primera entrega.
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "baseline_replay.json").write_text(json.dumps(reporte, indent=2) + "\n")
    print(json.dumps(reporte, indent=2))


if __name__ == "__main__":
    app()
