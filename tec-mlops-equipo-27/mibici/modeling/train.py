"""Baselines P1/P2/P3 con selección en validación y trazabilidad en MLflow."""

from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import subprocess
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
    EQUIPO,
    FIGURES_DIR,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    K_GRUPOS_CANDIDATOS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MUESTRA_SILHOUETTE,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
    SEED,
    UMBRAL_RIESGO,
    UMBRALES_CLASIFICACION,
    VENTANA_RIESGO_HORAS,
)
from mibici.preparar import preparar, verificar_datos

app = typer.Typer()


def informacion_codigo():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()

    commit = git("rev-parse", "HEAD")
    # Un run debe identificar exactamente código confirmado; el manifiesto sí puede
    # ser recién generado y su hash se registra de forma independiente.
    cambios = git("status", "--porcelain", "--", "mibici", "requirements.txt", "pyproject.toml")
    if cambios:
        raise ValueError("Confirma el código en Git antes de ejecutar baselines reproducibles.")
    return commit


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


def guardar_figura(nombre):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / nombre
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(path), artifact_path="figures")


def evaluar_p1(panel):
    # Una segunda referencia diaria permite una comparación de runs de la misma tarea.
    resultados = []
    for rezago in (168, 24):
        columna = f"salidas_rezago_{rezago}"
        partes = particiones(panel.dropna(subset=[columna]), panel.attrs["fin_exclusivo"])
        nombre = f"P1-baseline-rezago-{rezago}h"
        with mlflow.start_run(run_name=nombre) as run:
            registrar_contexto("P1", panel)
            mlflow.log_param("rezago_horas", rezago)
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
            diario = (
                partes["test"].groupby(partes["test"].periodo.dt.date)[["salidas", columna]].sum()
            )
            diario.plot(figsize=(9, 3), title=f"P1 · viajes diarios, prueba · rezago {rezago} h")
            guardar_figura(f"p1_rezago_{rezago}.png")
            resultados.append(
                {
                    "pregunta": "P1",
                    "modelo": nombre,
                    "run_id": run.info.run_id,
                    "metricas": metricas,
                }
            )
    return resultados


def evaluar_p2(panel):
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
    with mlflow.start_run(run_name="P2-baseline-frecuencia-estacion-hora") as run:
        registrar_contexto("P2", panel)
        mlflow.log_params(
            {
                "umbral_clasificacion": elegido["umbral"],
                "criterio_umbral": "max validation_f1; desempate recall, precision",
                "fallback": "frecuencia_global_train",
            }
        )
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
        mlflow.log_dict({"umbrales_validation": tabla}, "seleccion_umbral.json")
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
            "run_id": run.info.run_id,
            "metricas": metricas,
            "umbral": elegido["umbral"],
        }


def evaluar_p3(panel):
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
    candidatos = []
    modelos = {}
    for k in K_GRUPOS_CANDIDATOS:
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
    with mlflow.start_run(run_name="P3-baseline-kmeans-perfiles-train") as run:
        registrar_contexto("P3", panel)
        mlflow.log_params(
            {
                "k": elegido["k"],
                "n_init": 10,
                "escalado": "StandardScaler_train",
                "coordenadas_en_modelo": False,
                "seleccion_k": "max silhouette_train",
            }
        )
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
            "perfiles_medios_grupos.json",
        )
        mlflow.log_dict(
            {
                "candidatos": candidatos,
                "variables": columnas,
                "tamano_grupos": {
                    str(k): int(v) for k, v in perfiles.grupo.value_counts().items()
                },
            },
            "seleccion_grupos.json",
        )
        plt.figure(figsize=(7, 4))
        plt.scatter(perfiles.longitud, perfiles.latitud, c=perfiles.grupo, cmap="tab10", s=24)
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.title("P3 · grupos de patrones de uso (coordenadas solo para visualizar)")
        guardar_figura("p3_grupos.png")
        return {
            "pregunta": "P3",
            "modelo": "kmeans",
            "run_id": run.info.run_id,
            "metricas": metricas,
            "k": elegido["k"],
        }


def registrar_contexto(pregunta, panel):
    contexto = panel.attrs
    mlflow.set_tags(
        {
            "equipo": EQUIPO,
            "pregunta": pregunta,
            "git_commit": contexto["git_commit"],
            "tipo": "baseline",
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
        }
    )
    mlflow.log_dict(
        {p: version(p) for p in ("pandas", "numpy", "pyarrow", "scikit-learn", "mlflow")},
        "entorno.json",
    )


@app.command()
def main():
    commit = informacion_codigo()
    inicio = time.perf_counter()
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
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    resultados = [*evaluar_p1(panel), evaluar_p2(panel), evaluar_p3(panel)]
    reporte = {
        "estado": "ejecutado",
        "fecha_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "data_version": manifest["data_version"],
        "segundos": time.perf_counter() - inicio,
        "runs": resultados,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline_results.json").write_text(json.dumps(reporte, indent=2) + "\n")
    print(json.dumps(reporte, indent=2))


if __name__ == "__main__":
    app()
