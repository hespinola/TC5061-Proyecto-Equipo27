"""Candidatos, modelos completos y diagnósticos comparables con la entrega base."""

import time

import cloudpickle
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    PrecisionRecallDisplay,
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    mean_absolute_error,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici import features as f
from mibici.config import (
    FEATURES_P1,
    FEATURES_P2,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    SEED,
    SEMILLAS_ESTABILIDAD,
    UMBRALES_CLASIFICACION,
)
from mibici.modeling.estimators import FrequencyClassifier, LagRegressor, ThresholdClassifier


def guardar_modelo(model, X, resultado):
    """Un mismo modelo sirve para evaluación, sklearn y pyfunc, incluso el umbral P2."""
    example = X.head(5).astype(float)
    output = model.predict(example)
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=example,
        signature=mlflow.models.infer_signature(example, output),
        code_paths=[str(PROJ_ROOT / "mibici")],
        pip_requirements=[
            x
            for x in (PROJ_ROOT / "requirements.txt").read_text().splitlines()
            if x and not x.startswith(("#", "-e"))
        ],
    )
    resultado["model_uri"] = info.model_uri
    mlflow.set_tag("model_uri", info.model_uri)
    mlflow.log_dict(
        {
            "model_uri": info.model_uri,
            "input_columns": list(X.columns),
            "output": "grupo"
            if resultado["pregunta"] == "P3"
            else "alerta_0_1"
            if resultado["pregunta"] == "P2"
            else "viajes_por_hora",
        },
        "results/model_contract.json",
    )
    batch = X.head(1000).astype(float)
    model.predict(batch)  # calentamiento
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        model.predict(batch)
        timings.append((time.perf_counter() - start) * 1000 / len(batch))
    metrics = {
        "inference_ms_per_row": float(np.median(timings)),
        "model_bytes": len(cloudpickle.dumps(model)),
    }
    resultado["metricas"].update(metrics)
    mlflow.log_metrics(metrics)


def elegir_umbral(y, prob):
    from mibici.modeling.train import metricas_clasificacion

    table = [{"umbral": u, **metricas_clasificacion(y, prob, u)} for u in UMBRALES_CLASIFICACION]
    best = max(table, key=lambda r: (r["f1"], r["recall"], r["precision"], r["umbral"]))
    return best["umbral"], table


def diagnosticos_supervisados(partes, modelo, columnas, resultado, classification=False):
    from mibici.modeling.train import guardar_figura

    objetivo = "objetivo_en_riesgo" if classification else "salidas"
    weekly = []
    for split in ("validation", "test"):
        df = partes[split]
        pred = (
            modelo.predict_proba(df[columnas])[:, 1]
            if classification
            else modelo.predict(df[columnas])
        )
        records = pd.DataFrame(
            {"periodo": df.periodo.to_numpy(), "y": df[objetivo].to_numpy(), "pred": pred}
        )
        for week, group in records.groupby(records.periodo.dt.to_period("W")):
            if classification:
                from sklearn.metrics import average_precision_score

                if group.y.nunique() < 2:
                    continue
                score = average_precision_score(group.y.astype(int), group.pred)
            else:
                score = mean_absolute_error(group.y, group.pred)
            weekly.append({"split": split, "week": str(week), "score": float(score)})
        scores = [w["score"] for w in weekly if w["split"] == split]
        key = f"{split}_weekly_{'ap' if classification else 'mae'}_std"
        resultado["metricas"][key] = float(np.std(scores))
        if split != "test":
            continue
        if classification:
            matrix = confusion_matrix(
                records.y.astype(int), modelo.predict(df[columnas]), labels=[0, 1]
            )
            _fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center")
            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xlabel="Alerta predicha",
                ylabel="Proxy observado",
                title="P2 · matriz de confusión, prueba",
            )
            guardar_figura("p2_confusion.png")
            PrecisionRecallDisplay.from_predictions(records.y.astype(int), pred)
            plt.title("P2 · precisión-recall, prueba")
            guardar_figura("p2_precision_recall.png")
        else:
            daily = records.groupby(records.periodo.dt.date)[["y", "pred"]].sum()
            daily.plot(figsize=(9, 3), title="P1 · total diario, prueba")
            guardar_figura("p1_real_predicho.png")
            plt.figure(figsize=(7, 3))
            plt.hist(records.y.to_numpy() - pred, bins=80, range=(-10, 10))
            plt.xlabel("Observado - predicho (viajes / estación / hora)")
            plt.title("P1 · residuales, prueba; rango visual [-10, 10]")
            guardar_figura("p1_residuales.png")
    mlflow.log_dict(
        {"weekly_scores": weekly, "note": "Dispersión descriptiva; no IC iid."},
        "results/estabilidad_temporal.json",
    )


def evaluar_candidato(panel, tarea):
    if tarea.pregunta == "p3":
        return evaluar_grupos(panel, tarea)
    from mibici.modeling.train import metricas_clasificacion, particiones, registrar_dataset

    classification = tarea.pregunta == "p2"
    columns = list(FEATURES_P2 if classification else FEATURES_P1)
    target = "objetivo_en_riesgo" if classification else "salidas"
    # Misma elegibilidad semanal y mismas filas de validación/prueba que el baseline.
    required = ["salidas_rezago_168"] + ([target] if classification else [])
    parts = particiones(
        panel.dropna(subset=required), panel.attrs["fin_exclusivo"], p2=classification
    )
    for split, df in parts.items():
        dataset = df[columns].astype(float).assign(**{target: df[target].astype(float)})
        registrar_dataset(dataset, panel, split, target)
    hp = tarea.hiperparametros
    cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
    model = cls(
        max_iter=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        learning_rate=float(hp["learning_rate"]),
        random_state=SEED,
        early_stopping=False,
        categorical_features=None,
    )
    start = time.perf_counter()
    model.fit(
        parts["train"][columns], parts["train"][target].astype(int if classification else float)
    )
    metrics = {"fit_seconds": time.perf_counter() - start, "train_filas": len(parts["train"])}
    result = {"pregunta": tarea.pregunta.upper(), "modelo": tarea.modelo, "metricas": metrics}
    if classification:
        threshold, table = elegir_umbral(
            parts["validation"][target], model.predict_proba(parts["validation"][columns])[:, 1]
        )
        model = ThresholdClassifier(model, threshold)
        result["umbral"] = threshold
        mlflow.log_param("umbral_clasificacion", threshold)
        mlflow.log_dict({"umbrales_validation": table}, "results/seleccion_umbral.json")
    for split in ("validation", "test"):
        df = parts[split]
        metrics[f"{split}_filas"] = len(df)
        if classification:
            prob = model.predict_proba(df[columns])[:, 1]
            values = metricas_clasificacion(df[target], prob, threshold)
        else:
            pred = model.predict(df[columns])
            values = {
                "mae": mean_absolute_error(df[target], pred),
                "rmse": root_mean_squared_error(df[target], pred),
            }
        metrics.update({f"{split}_{k}": float(v) for k, v in values.items()})
    mlflow.set_tag("mlflow.runName", f"{tarea.pregunta.upper()}-{tarea.modelo}-{hp}")
    diagnosticos_supervisados(parts, model, columns, result, classification)
    guardar_modelo(model, parts["train"][columns], result)
    mlflow.log_metrics(metrics)
    return result


def perfiles_periodo(panel, fin=None):
    nom = pd.read_parquet(PROCESSED_DATA_DIR / "nomenclatura.parquet")
    profiles = f.construir_perfil_estaciones(panel, nom)
    cols = [c for c in profiles if c.startswith("perfil_")] + [
        "salidas_promedio_dia",
        "balance_llegadas_salidas",
    ]
    valid = (profiles.salidas_promedio_dia > 0) & profiles[cols].notna().all(axis=1)
    return profiles.loc[valid].set_index("estacion_id"), cols, int((~valid).sum())


def cluster_scores(X, labels):
    unique = len(np.unique(labels))
    if not 1 < unique < len(X):
        # No esconder una partición colapsada: la selección la rechaza.
        return {"silhouette": -1.0, "davies_bouldin": 1e12, "collapsed": 1}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "collapsed": 0,
    }


def evaluar_grupos(panel, tarea, resultado=None):
    from mibici.modeling.train import guardar_figura, particiones, registrar_dataset

    parts = particiones(panel, panel.attrs["fin_exclusivo"])
    profiles, cols, excluded = perfiles_periodo(parts["train"])
    k = int(resultado["k"] if resultado else tarea.hiperparametros["n_clusters"])
    steps = [("scale", StandardScaler())]
    if tarea.modelo == "pca_kmeans":
        steps.append(
            (
                "pca",
                PCA(n_components=int(tarea.hiperparametros["n_components"]), svd_solver="full"),
            )
        )
    steps.append(("cluster", KMeans(n_clusters=k, random_state=SEED, n_init=10)))
    model = Pipeline(steps)
    start = time.perf_counter()
    model.fit(profiles[cols])
    metrics = {
        "fit_seconds": time.perf_counter() - start,
        "estaciones_agrupadas": len(profiles),
        "estaciones_excluidas": excluded,
    }
    labels = model.predict(profiles[cols])
    scaled = model.named_steps["scale"].transform(profiles[cols])
    # Misma geometría de 50 variables escaladas para todos, aun cuando se ajuste PCA.
    metrics.update(
        {f"train_{key}": value for key, value in cluster_scores(scaled, labels).items()}
    )
    seed_ari = []
    for seed in SEMILLAS_ESTABILIDAD[1:]:
        replica = clone(model).set_params(cluster__random_state=seed).fit(profiles[cols])
        seed_ari.append(float(adjusted_rand_score(labels, replica.predict(profiles[cols]))))
    metrics["seed_ari_mean"] = float(np.mean(seed_ari))
    metrics["seed_ari_min"] = float(np.min(seed_ari))
    for split in ("train", "validation", "test"):
        p, _, _ = (
            perfiles_periodo(parts[split]) if split != "train" else (profiles, cols, excluded)
        )
        p = p.loc[p.index.intersection(profiles.index)]
        registrar_dataset(p[cols].astype(float), panel, split)
        pred = model.predict(p[cols])
        metrics[f"{split}_estaciones"] = len(p)
        if split != "train":
            scores = cluster_scores(model.named_steps["scale"].transform(p[cols]), pred)
            metrics.update({f"{split}_{key}": value for key, value in scores.items()})
            train_labels = pd.Series(labels, index=profiles.index).loc[p.index]
            metrics[f"{split}_temporal_ari"] = float(adjusted_rand_score(train_labels, pred))
    if resultado is None:
        resultado = {"pregunta": "P3", "modelo": tarea.modelo, "metricas": {}, "k": k}
    resultado["metricas"].update(metrics)
    profiles["grupo"] = labels
    sizes = {str(g): int(n) for g, n in profiles.grupo.value_counts().items()}
    metrics["min_cluster_size"] = min(sizes.values())
    mlflow.log_dict(
        {
            "variables": cols,
            "tamano_grupos": sizes,
            "seed_ari": seed_ari,
            "geometry": "50 variables; StandardScaler ajustado en train",
        },
        "results/diagnostico_grupos.json",
    )
    means = profiles.groupby("grupo")[cols].mean()
    mlflow.log_dict(means.to_dict("index"), "results/perfiles_medios_grupos.json")
    plt.figure(figsize=(7, 4))
    plt.scatter(profiles.longitud, profiles.latitud, c=labels, cmap="tab10", s=24)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(f"P3 · {tarea.modelo}, k={k}; coordenadas solo para visualizar")
    guardar_figura("p3_grupos.png")
    means[[c for c in cols if c.startswith("perfil_lab_")]].T.plot(figsize=(9, 4))
    plt.title("P3 · perfil de salidas entre semana por grupo")
    plt.xlabel("Hora")
    guardar_figura("p3_perfiles.png")
    mlflow.log_params({"cluster_k": k, "cluster_n_init": 10})
    if tarea.modelo != "baseline":
        mlflow.set_tag("mlflow.runName", f"P3-{tarea.modelo}-k{k}")
    guardar_modelo(model, profiles[cols], resultado)
    mlflow.log_metrics(resultado["metricas"])
    return resultado


def completar_baseline(panel, tarea, resultado):
    """Conserva métricas previas y añade el estimador y diagnósticos desplegables."""
    from mibici.modeling.train import particiones

    if tarea.pregunta == "p3":
        evaluar_grupos(panel, tarea, resultado)
        return
    classification = tarea.pregunta == "p2"
    cols = (
        ["estacion_id", "hora_del_dia"]
        if classification
        else [f"salidas_rezago_{tarea.hiperparametros['rezago_horas']}"]
    )
    required = ["salidas_rezago_168", "objetivo_en_riesgo"] if classification else cols
    parts = particiones(
        panel.dropna(subset=required), panel.attrs["fin_exclusivo"], p2=classification
    )
    start = time.perf_counter()
    if classification:
        estimator = FrequencyClassifier().fit(
            parts["train"][cols], parts["train"].objetivo_en_riesgo
        )
        model = ThresholdClassifier(estimator, resultado["umbral"])
    else:
        model = LagRegressor(cols[0]).fit(parts["train"][cols])
    resultado["metricas"]["fit_seconds"] = time.perf_counter() - start
    diagnosticos_supervisados(parts, model, cols, resultado, classification)
    guardar_modelo(model, parts["train"][cols], resultado)
    mlflow.log_metrics(resultado["metricas"])
