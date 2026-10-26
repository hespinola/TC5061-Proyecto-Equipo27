"""Reejecuta baselines y búsquedas desde un único MLflow Project, sin notebooks."""

from datetime import UTC, datetime
import json
import math
import os

import mlflow
import typer

from mibici.config import (
    BUSQUEDAS,
    EQUIPO,
    MLFLOW_EXPERIMENT_NAME,
    PREGUNTAS,
    REPORTS_DIR,
    RESULTADOS_EXPERIMENTOS_PATH,
    REZAGOS_HORAS,
)
from mibici.modeling.train import (
    Tarea,
    cargar_panel,
    ejecutar_tarea,
    informacion_codigo,
    preparar_experimento,
    resolver_hiperparametros,
)

app = typer.Typer()


def comprobar_referencia(resultados, data_version):
    """La repetición del baseline debe reproducir métricas y población anteriores."""
    reference = json.loads((REPORTS_DIR / "baseline_results.json").read_text())
    if reference["data_version"] != data_version:
        raise ValueError("La versión de datos difiere de la primera entrega.")
    checks = []
    for result in resultados:
        if result["stage_of_work"] != "baseline":
            continue
        old = next(
            r
            for r in reference["runs"]
            if r["pregunta"] == result["pregunta"]
            and (r["modelo"] == result["modelo"] or result["pregunta"] != "P1")
        )
        for key, value in old["metricas"].items():
            actual = result["metricas"][key]
            if not math.isclose(value, actual, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(
                    f"Baseline incompatible: {result['pregunta']} {key}: {value} != {actual}"
                )
        checks.append(
            {"run_id": result["run_id"], "original_run_id": old["run_id"], "metrics_match": True}
        )
    return checks


@app.command()
def main(question: str = "all"):
    questions = PREGUNTAS if question == "all" else (question,)
    if any(q not in PREGUNTAS for q in questions):
        raise ValueError("question debe ser p1, p2, p3 o all.")
    commit = informacion_codigo()
    preparar_experimento()
    panel = cargar_panel(commit)
    results = []
    with mlflow.start_run() as project_run:
        # start_run retoma MLFLOW_RUN_ID una vez; los hijos deben crear IDs nuevos.
        os.environ.pop("MLFLOW_RUN_ID", None)
        common = {
            "equipo": EQUIPO,
            "git_commit": commit,
            "data_version": panel.attrs["data_version"],
        }
        mlflow.set_tags(
            {
                **common,
                "mlflow.runName": f"entrega2-{question}",
                "question": question,
                "stage_of_work": "experiment",
                "model_family": "suite",
            }
        )
        for q in questions:
            baselines = (
                [
                    Tarea(q, "baseline", {"rezago_horas": h})
                    for h in sorted(REZAGOS_HORAS, reverse=True)
                ]
                if q == "p1"
                else [Tarea(q, "baseline", resolver_hiperparametros(q, "baseline", {}))]
            )
            for task in baselines:
                with mlflow.start_run(nested=True):
                    result = ejecutar_tarea(task, panel, "baseline")
                    result.update(stage_of_work="baseline", hp=task.hiperparametros)
                    results.append(result)
            with mlflow.start_run(nested=True) as search:
                mlflow.set_tags(
                    {
                        **common,
                        "mlflow.runName": f"search-{q}",
                        "question": q,
                        "stage_of_work": "search",
                        "model_family": "grid",
                    }
                )
                mlflow.log_dict(
                    {"combinations": BUSQUEDAS[q], "selection_uses_test": False},
                    "config/search_space.json",
                )
                for model, hp in BUSQUEDAS[q]:
                    with mlflow.start_run(nested=True):
                        task = Tarea(q, model, hp)
                        result = ejecutar_tarea(task, panel, "candidate")
                        result.update(
                            stage_of_work="candidate", hp=hp, search_run_id=search.info.run_id
                        )
                        results.append(result)
        parity = comprobar_referencia(results, panel.attrs["data_version"])
        report = {
            "estado": "ejecutado",
            "fecha_utc": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "data_version": panel.attrs["data_version"],
            "experiment": MLFLOW_EXPERIMENT_NAME,
            "project_run_id": project_run.info.run_id,
            "baseline_parity": parity,
            "runs": results,
        }
        mlflow.log_dict(report, "results/experiment_results.json")
    RESULTADOS_EXPERIMENTOS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
