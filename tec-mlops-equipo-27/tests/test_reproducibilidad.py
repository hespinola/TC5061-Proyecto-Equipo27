"""Reproducibilidad del entrenamiento: misma semilla, mismas métricas y predicciones.

Ejecuta dos veces el código real de entrenamiento de ``mibici.modeling.candidates``
(HistGradientBoosting para P1/P2 y PCA + K-Means para P3) sobre un panel sintético
pequeño, y compara métricas y predicciones de los dos modelos resultantes.

Es aislada del entorno: el tracking de MLflow usa una base SQLite temporal (``tmp_path``),
por lo que no requiere servidor MLflow, descargas de MiBici ni escribe en el repositorio.

Ejecución (desde ``tec-mlops-equipo-27/``)::

    python -m pytest tests/test_reproducibilidad.py -v
"""

import matplotlib

matplotlib.use("Agg")

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici.config import FEATURES_P1, FEATURES_P2, OBSERVACION_FIN, OBSERVACION_INICIO
from mibici.modeling import candidates
from mibici.modeling.train import Tarea, particiones

# Costo de ejecución (tiempo y tamaño serializado): varía entre corridas aunque el
# resultado sea idéntico, por eso no entra en la comparación de calidad.
METRICAS_DE_COSTO = {"fit_seconds", "inference_ms_per_row", "model_bytes"}
N_ESTACIONES = 30
FILAS_POR_ESTACION = 400

CASOS = [
    pytest.param("p1", "hgb", {"n_estimators": 8, "max_depth": 3, "learning_rate": 0.1}, id="P1-hgb"),
    pytest.param("p2", "hgb", {"n_estimators": 8, "max_depth": 3, "learning_rate": 0.1}, id="P2-hgb"),
    pytest.param("p3", "pca_kmeans", {"n_clusters": 3, "n_components": 5}, id="P3-pca-kmeans"),
]


def panel_sintetico() -> pd.DataFrame:
    """Panel estación x hora con las columnas que consumen los entrenadores.

    Usa una semilla de datos fija e independiente de la semilla del modelo, de modo que
    las dos ejecuciones comparadas entrenan sobre exactamente los mismos datos.
    """
    rng = np.random.default_rng(2026)
    horas = pd.date_range(OBSERVACION_INICIO, OBSERVACION_FIN, freq="h", inclusive="left")
    bloques = []
    for estacion in range(1, N_ESTACIONES + 1):
        elegidas = np.sort(rng.choice(len(horas), size=FILAS_POR_ESTACION, replace=False))
        bloques.append(pd.DataFrame({"estacion_id": estacion, "periodo": horas[elegidas]}))
    panel = pd.concat(bloques, ignore_index=True)

    hora = panel.periodo.dt.hour
    dia = panel.periodo.dt.dayofweek
    panel["hora_del_dia"] = hora
    panel["dia_semana"] = dia
    panel["es_fin_de_semana"] = (dia >= 5).astype(int)
    panel["mes"] = panel.periodo.dt.month
    panel["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    panel["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    panel["dia_sin"] = np.sin(2 * np.pi * dia / 7)
    panel["dia_cos"] = np.cos(2 * np.pi * dia / 7)

    # Demanda con perfil por estación y hora, para que los modelos aprendan algo real.
    tasa = (1 + panel.estacion_id % 5) * (1 + 2 * np.exp(-((hora - 8) ** 2) / 8))
    panel["salidas"] = rng.poisson(tasa)
    panel["llegadas"] = rng.poisson(tasa)
    for base in ("salidas", "llegadas"):
        for h in (24, 168):
            panel[f"{base}_rezago_{h}"] = rng.poisson(tasa).astype(float)
            panel[f"{base}_media_{h}"] = tasa + rng.normal(0, 0.3, len(panel))
    panel["capacidad"] = 15.0 + 5 * (panel.estacion_id % 3)
    panel["flujo_acumulado_dia_rel"] = rng.normal(0, 0.2, len(panel))
    panel["objetivo_en_riesgo"] = (rng.random(len(panel)) < 0.15 + 0.1 * (tasa > 4)).astype(int)
    panel["objetivo_fin_periodo"] = panel.periodo + pd.Timedelta(hours=3)
    panel.attrs = {"data_version": "sintetico-v1", "fin_exclusivo": OBSERVACION_FIN}
    return panel


@pytest.fixture
def entorno_aislado(tmp_path, monkeypatch):
    """MLflow SQLite temporal y nomenclatura sintética; nada se escribe en el repositorio."""
    for variable in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_EXPERIMENT_ID"):
        monkeypatch.delenv(variable, raising=False)
    anterior = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    experimento = mlflow.create_experiment(
        "reproducibilidad", artifact_location=(tmp_path / "artefactos").as_uri()
    )
    mlflow.set_experiment(experiment_id=experimento)

    rng = np.random.default_rng(7)
    pd.DataFrame(
        {
            "id": np.arange(1, N_ESTACIONES + 1),
            "latitude": 20.67 + rng.normal(0, 0.02, N_ESTACIONES),
            "longitude": -103.35 + rng.normal(0, 0.02, N_ESTACIONES),
            "dpcapacity": 15 + 5 * (np.arange(1, N_ESTACIONES + 1) % 3),
        }
    ).to_parquet(tmp_path / "nomenclatura.parquet", index=False)
    monkeypatch.setattr(candidates, "PROCESSED_DATA_DIR", tmp_path)
    yield
    mlflow.end_run()
    mlflow.set_tracking_uri(anterior)


def entrenar(tarea: Tarea, panel: pd.DataFrame) -> dict:
    """Una ejecución completa de entrenamiento, en su propio run de MLflow."""
    with mlflow.start_run():
        return candidates.evaluar_candidato(panel, tarea)


def entradas_de_prediccion(pregunta: str, panel: pd.DataFrame) -> pd.DataFrame:
    """Mismas filas de entrada para evaluar los modelos de ambas ejecuciones."""
    if pregunta == "p3":
        entrenamiento = particiones(panel, panel.attrs["fin_exclusivo"])["train"]
        perfiles, columnas, _ = candidates.perfiles_periodo(entrenamiento)
        return perfiles[columnas].astype(float)
    columnas = FEATURES_P2 if pregunta == "p2" else FEATURES_P1
    return panel[list(columnas)].head(500).astype(float)


def predicciones(modelo, X: pd.DataFrame) -> dict[str, np.ndarray]:
    salidas = {"predict": np.asarray(modelo.predict(X))}
    if hasattr(modelo, "predict_proba"):
        salidas["predict_proba"] = np.asarray(modelo.predict_proba(X))
    return salidas


@pytest.mark.parametrize("semilla", [42, 7])
@pytest.mark.parametrize(("pregunta", "modelo", "hiperparametros"), CASOS)
def test_misma_semilla_reproduce_metricas_y_predicciones(
    entorno_aislado, monkeypatch, pregunta, modelo, hiperparametros, semilla
):
    monkeypatch.setattr(candidates, "SEED", semilla)
    panel = panel_sintetico()
    tarea = Tarea(pregunta, modelo, hiperparametros)

    primera = entrenar(tarea, panel)
    segunda = entrenar(tarea, panel)

    # Son dos entrenamientos independientes, no el mismo objeto reutilizado.
    assert primera["model_uri"] != segunda["model_uri"]

    # Métricas de calidad: mismas llaves y mismos valores (se excluye solo el costo).
    m1 = {k: v for k, v in primera["metricas"].items() if k not in METRICAS_DE_COSTO}
    m2 = {k: v for k, v in segunda["metricas"].items() if k not in METRICAS_DE_COSTO}
    assert m1.keys() == m2.keys()
    assert m1, "El entrenamiento no produjo métricas que comparar."
    for nombre in m1:
        assert m1[nombre] == pytest.approx(m2[nombre], rel=1e-12, abs=0, nan_ok=True), nombre

    # Predicciones: los modelos guardados responden igual sobre las mismas entradas.
    X = entradas_de_prediccion(pregunta, panel)
    p1 = predicciones(mlflow.sklearn.load_model(primera["model_uri"]), X)
    p2 = predicciones(mlflow.sklearn.load_model(segunda["model_uri"]), X)
    assert p1.keys() == p2.keys()
    for nombre in p1:
        assert np.isfinite(p1[nombre]).all(), nombre
        if p1[nombre].dtype.kind in "iu":
            np.testing.assert_array_equal(p1[nombre], p2[nombre], err_msg=nombre)
        else:
            np.testing.assert_allclose(p1[nombre], p2[nombre], rtol=1e-12, atol=0, err_msg=nombre)