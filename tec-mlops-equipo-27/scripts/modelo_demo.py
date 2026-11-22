"""Genera un modelo P1 de DEMOSTRACIÓN en ``modelo_p1/`` sin datos ni servidor de MLflow.

    python scripts/modelo_demo.py --destino modelo_p1

Sirve para construir la imagen en un clon limpio o en GitHub Actions, donde no existe
el Model Registry del equipo. Entrena un ``HistGradientBoostingRegressor`` pequeño con
datos sintéticos, lo guarda con ``guardar_modelo`` (la misma función del entrenamiento,
así la firma coincide con ``FEATURES_P1``), lo registra en un registry SQLite temporal
y lo copia con ``mibici.api.descargar``. Sus predicciones NO son las del modelo real:
``modelo_info.json`` lo marca con el alias ``demo``. Para servir el modelo del equipo,
usar ``make modelo`` con el registry correspondiente.
"""

import os
from pathlib import Path
import tempfile

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import typer

from mibici import features as f
from mibici.api import descargar
from mibici.api.modelo import NOMBRE_POR_DEFECTO
from mibici.config import FEATURES_P1
from mibici.modeling.candidates import guardar_modelo

ALIAS_DEMO = "demo"

app = typer.Typer()


def datos_sinteticos(n: int = 1500, semilla: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(semilla)
    X = f.agregar_variables_calendario(
        pd.DataFrame({f.COL_PERIODO: pd.date_range("2026-01-01", periods=n, freq="h")})
    )
    for col in FEATURES_P1:
        if col not in X:
            X[col] = rng.poisson(2, n).astype(float)
    X = X[list(FEATURES_P1)].astype(float)
    y = (
        0.6 * X["salidas_rezago_24"]
        + 0.4 * X["salidas_media_168"]
        + 3 * X["hora_del_dia"].between(7, 9)
    )
    return X, y


@app.command()
def main(destino: Path = Path("modelo_p1")):
    with tempfile.TemporaryDirectory() as raiz:
        uri = f"sqlite:///{(Path(raiz) / 'mlflow.db').as_posix()}"
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ.pop("MODELO_VERSION", None)
        os.environ.pop("MODELO_RUTA", None)
        os.environ["MODELO_NOMBRE"] = NOMBRE_POR_DEFECTO
        os.environ["MODELO_ALIAS"] = ALIAS_DEMO
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(
            experiment_id=mlflow.create_experiment(
                "modelo-demo", artifact_location=(Path(raiz) / "artefactos").as_uri()
            )
        )
        X, y = datos_sinteticos()
        modelo = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)
        with mlflow.start_run(run_name="modelo-demo-ci"):
            resultado = {"pregunta": "P1", "metricas": {}}
            guardar_modelo(modelo, X, resultado)
        version = mlflow.register_model(resultado["model_uri"], NOMBRE_POR_DEFECTO).version
        mlflow.MlflowClient().set_registered_model_alias(NOMBRE_POR_DEFECTO, ALIAS_DEMO, version)
        # Misma copia y verificación de firma que se usa con el registry real
        descargar.main(destino=destino)


if __name__ == "__main__":
    app()
