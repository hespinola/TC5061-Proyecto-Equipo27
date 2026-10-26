"""Selección en validación, promoción por alias e inferencia con el nombre público."""

from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    EQUIPO,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    METRICA_SELECCION,
    MLFLOW_EXPERIMENT_NAME,
    PREGUNTAS,
    REPORTS_DIR,
    RESULTADOS_EXPERIMENTOS_PATH,
    TOLERANCIA_SELECCION,
    VALOR_DESDE_CONFIG,
)
from mibici.modeling.train import informacion_codigo, preparar_experimento

app = typer.Typer()
SUFIJOS = {"p1": "demanda", "p2": "riesgo", "p3": "grupos"}
DESCRIPCIONES = {
    "p1": "Predice viajes que comienzan por estación y hora al inicio de t; usa historia anterior a t.",
    "p2": "Prioriza riesgo de magnitud del flujo neto de 3 h >=50% de capacidad. Es un proxy, no inventario.",
    "p3": "Agrupa perfiles horarios de estaciones; escala y proyección ajustadas exclusivamente en train.",
}


def seleccionar(runs, question):
    """No consulta ninguna métrica test; incluye el baseline si continúa siendo mejor."""
    key = METRICA_SELECCION[question]
    eligible = [
        r
        for r in runs
        if r["pregunta"].lower() == question
        and math.isfinite(r["metricas"].get(key, float("nan")))
    ]
    if question == "p3":
        eligible = [r for r in eligible if not r["metricas"].get("validation_collapsed", 1)]
    if not eligible:
        raise ValueError(f"No hay modelos elegibles para {question}.")
    minimize = question == "p1"
    best = (min if minimize else max)(r["metricas"][key] for r in eligible)
    tol = TOLERANCIA_SELECCION[question] * (abs(best) if question != "p3" else 1)
    pool = [r for r in eligible if abs(r["metricas"][key] - best) <= tol + 1e-12]
    if question == "p1":
        tie = lambda r: (
            r["metricas"]["validation_weekly_mae_std"],
            r["metricas"]["model_bytes"],
            r["run_id"],
        )
    elif question == "p2":
        tie = lambda r: (
            -r["metricas"]["validation_recall"],
            -r["metricas"]["validation_f1"],
            r["metricas"]["model_bytes"],
            r["run_id"],
        )
    else:
        tie = lambda r: (
            -r["metricas"]["seed_ari_min"],
            -r["metricas"]["validation_temporal_ari"],
            r["metricas"]["model_bytes"],
            r["run_id"],
        )
    chosen = min(pool, key=tie)
    return chosen, {
        "metric": key,
        "best_validation_value": best,
        "tolerance": tol,
        "near_tie_run_ids": [r["run_id"] for r in pool],
        "test_used_for_selection": False,
        "tie_break": {
            "p1": "menor dispersión MAE semanal; menor tamaño",
            "p2": "mayor recall; mayor F1; menor tamaño",
            "p3": "mayor ARI mínimo entre semillas; ARI temporal; menor tamaño",
        }[question],
    }


def ejemplo_modelo(uri):
    model = mlflow.models.Model.load(uri)
    if not model.signature or not model.saved_input_example_info:
        raise ValueError("Modelo sin firma o ejemplo de entrada.")
    root = Path(mlflow.artifacts.download_artifacts(uri))
    example = json.loads((root / model.saved_input_example_info["artifact_path"]).read_text())
    return pd.DataFrame(example["data"], columns=example["columns"])


def comprobar_corrida(client, row, report):
    run = client.get_run(row["run_id"])
    experiment = client.get_experiment(run.info.experiment_id)
    expected = {
        "equipo": EQUIPO,
        "question": row["pregunta"].lower(),
        "git_commit": report["git_commit"],
        "data_version": report["data_version"],
    }
    if run.info.status != "FINISHED" or experiment.name != MLFLOW_EXPERIMENT_NAME:
        raise ValueError("La corrida debe estar terminada en el experimento del equipo.")
    if any(run.data.tags.get(k) != v for k, v in expected.items()):
        raise ValueError("El linaje de la corrida no coincide con el resumen.")
    if run.data.tags.get("model_uri") != row["model_uri"]:
        raise ValueError("El URI del modelo no corresponde al registrado por la corrida.")
    for key, value in row["metricas"].items():
        if not math.isclose(value, run.data.metrics.get(key, float("nan")), rel_tol=1e-10):
            raise ValueError(f"Métrica alterada en el resumen: {key}")
    params = run.data.params
    if (
        params.get("fin_train") != FIN_ENTRENAMIENTO
        or params.get("fin_validation") != FIN_VALIDACION
    ):
        raise ValueError("Los cortes temporales no coinciden con config.py.")
    # Comprobar fuentes y entradas, antes de crear o promover cualquier versión.
    paths = {x.path for x in client.list_artifacts(row["run_id"])}
    if not {"config", "data", "plots", "results"} <= paths or not run.inputs.dataset_inputs:
        raise ValueError("Faltan artefactos o dataset inputs.")
    if row["stage_of_work"] == "candidate":
        parent = run.data.tags.get("mlflow.parentRunId")
        if not parent or client.get_run(parent).data.tags.get("stage_of_work") != "search":
            raise ValueError("La candidata no pertenece a una búsqueda anidada.")
    return run


def promover(client, row, report, criterion, registry_commit):
    question = row["pregunta"].lower()
    name = f"mibici-{EQUIPO}-{question}-{SUFIJOS[question]}"
    existing = [
        v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == row["run_id"]
    ]
    if existing:
        version = existing[0]
    else:
        version = mlflow.register_model(row["model_uri"], name, await_registration_for=120)
    number = str(version.version)
    lineage = {
        "equipo": EQUIPO,
        "question": question,
        "git_commit": report["git_commit"],
        "data_version": report["data_version"],
        "source_run_id": row["run_id"],
        "source_experiment": MLFLOW_EXPERIMENT_NAME,
        "registry_commit": registry_commit,
        "lifecycle_scheme": "alias",
        "production_alias": "champion",
    }
    client.update_registered_model(
        name,
        description=DESCRIPCIONES[question]
        + " Ciclo de vida por alias champion; evaluación histórica local.",
    )
    for key, value in lineage.items():
        client.set_model_version_tag(name, number, key, value)
    description = {
        "purpose": DESCRIPCIONES[question],
        **lineage,
        "selection": criterion,
        "hyperparameters": row["hp"],
    }
    client.update_model_version(
        name, number, description=json.dumps(description, ensure_ascii=False, indent=2)
    )
    X = ejemplo_modelo(row["model_uri"])
    expected = mlflow.pyfunc.load_model(row["model_uri"]).predict(X)
    # Validar versión antes de mover el alias vigente.
    actual = mlflow.pyfunc.load_model(f"models:/{name}/{number}").predict(X)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    try:
        previous = client.get_model_version_by_alias(name, "champion").version
    except mlflow.exceptions.MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        previous = None
    client.set_registered_model_alias(name, "champion", number)
    uri = f"models:/{name}@champion"
    try:
        np.testing.assert_allclose(
            mlflow.pyfunc.load_model(uri).predict(X), expected, rtol=1e-12, atol=1e-12
        )
    except Exception:
        if previous is not None:
            client.set_registered_model_alias(name, "champion", previous)
        else:
            client.delete_registered_model_alias(name, "champion")
        raise
    client.set_model_version_tag(name, number, "validation_status", "passed")
    return {
        "name": name,
        "version": number,
        "alias": "champion",
        "uri": uri,
        "run_id": row["run_id"],
        "model_uri": row["model_uri"],
        "source_commit": report["git_commit"],
        "data_version": report["data_version"],
        "load_by_alias_verified": True,
        "prediction_parity_verified": True,
        "example_rows": len(X),
        "selection": criterion,
    }


@app.command()
def main(results: str = VALOR_DESDE_CONFIG):
    path = RESULTADOS_EXPERIMENTOS_PATH if results == VALOR_DESDE_CONFIG else Path(results)
    report = json.loads(path.read_text())
    commit = informacion_codigo()
    preparar_experimento()
    client = mlflow.MlflowClient()
    # Validar TODAS las preguntas antes de modificar el Registry.
    selections = []
    if report.get("estado") != "ejecutado":
        raise ValueError("La experimentación no terminó correctamente.")
    for q in PREGUNTAS:
        rows = [r for r in report["runs"] if r["pregunta"].lower() == q]
        if sum(r["stage_of_work"] == "candidate" for r in rows) < 3 or not any(
            r["stage_of_work"] == "baseline" for r in rows
        ):
            raise ValueError(f"Faltan baseline o tres candidatas de {q}.")
        for row in rows:
            comprobar_corrida(client, row, report)
        chosen, criterion = seleccionar(rows, q)
        ejemplo_modelo(chosen["model_uri"])
        selections.append((chosen, criterion))
    with mlflow.start_run() as run:
        os.environ.pop("MLFLOW_RUN_ID", None)
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "stage_of_work": "registry",
                "git_commit": commit,
                "data_version": report["data_version"],
                "mlflow.runName": "promote-champions",
            }
        )
        promoted = [
            promover(client, row, report, criterion, commit) for row, criterion in selections
        ]
        output = {
            "fecha_utc": datetime.now(UTC).isoformat(),
            "registry_commit": commit,
            "source_commit": report["git_commit"],
            "data_version": report["data_version"],
            "registry_run_id": run.info.run_id,
            "models": promoted,
        }
        mlflow.log_dict(output, "results/registry_results.json")
    (REPORTS_DIR / "registry_results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
