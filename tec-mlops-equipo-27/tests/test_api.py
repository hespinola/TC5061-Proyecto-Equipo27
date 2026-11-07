"""API de predicción de demanda (mibici/api).

Registra un modelo P1 pequeño con ``guardar_modelo`` (la misma función del
entrenamiento) en un Model Registry SQLite temporal, levanta la API con
``TestClient`` y verifica carga por alias y por versión, carga desde una copia
local, paridad con el modelo, validación de entradas y salidas, y errores.
No requiere datos descargados ni servidor de MLflow:

    python -m pytest tests/test_api.py -v
"""

import json
import math

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from typer.testing import CliRunner

from mibici import features as f
from mibici.api import descargar
from mibici.api.app import app
from mibici.api.esquemas import EJEMPLO_INSTANCIA, MAX_INSTANCIAS, InfoModelo, InstanciaDemanda
from mibici.api.modelo import (
    NOMBRE_POR_DEFECTO,
    Ajustes,
    ErrorModelo,
    ModeloCargado,
    Servicio,
    cargar,
    construir_variables,
    predecir,
)
from mibici.config import FEATURES_P1
from mibici.modeling.candidates import guardar_modelo

VARIABLES_ENTORNO = ("MODELO_NOMBRE", "MODELO_VERSION", "MODELO_ALIAS", "MODELO_RUTA")


def datos_entrenamiento(n=1500, semilla=0):
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


def registrar(X, y, nombre=NOMBRE_POR_DEFECTO, max_iter=20):
    modelo = HistGradientBoostingRegressor(max_iter=max_iter, random_state=0).fit(X, y)
    with mlflow.start_run():
        resultado = {"pregunta": "P1", "metricas": {}}
        guardar_modelo(modelo, X, resultado)
    # El registry SQL devuelve la versión como entero; la API la reporta como texto.
    return str(mlflow.register_model(resultado["model_uri"], nombre).version)


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Registry con dos versiones de P1; ``champion`` apunta a la 1."""
    raiz = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{(raiz / 'mlflow.db').as_posix()}"
    anterior = mlflow.get_tracking_uri()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(
            experiment_id=mlflow.create_experiment("api", artifact_location=(raiz / "a").as_uri())
        )
        X, y = datos_entrenamiento()
        v1 = registrar(X, y)
        v2 = registrar(X, y, max_iter=5)
        mlflow.MlflowClient().set_registered_model_alias(NOMBRE_POR_DEFECTO, "champion", v1)
        yield {"uri": uri, "raiz": raiz, "v1": v1, "v2": v2, "X": X}
    mlflow.set_tracking_uri(anterior)


@pytest.fixture
def entorno(registry, monkeypatch):
    for variable in VARIABLES_ENTORNO:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", registry["uri"])
    mlflow.set_tracking_uri(registry["uri"])
    return monkeypatch


@pytest.fixture
def cliente(entorno):
    with TestClient(app) as c:
        yield c


def instancia(**cambios):
    return {**EJEMPLO_INSTANCIA, **cambios}


def solicitud(*instancias):
    return {"instancias": list(instancias) or [instancia()]}


# ---------------------------------------------------------------------------
# Recuperación del modelo desde el Model Registry
# ---------------------------------------------------------------------------
def test_carga_por_alias_reporta_la_version_resuelta(cliente, registry):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    modelo = respuesta.json()["modelo"]
    assert modelo == {
        "nombre": NOMBRE_POR_DEFECTO,
        "version": registry["v1"],
        "alias": "champion",
        "uri": f"models:/{NOMBRE_POR_DEFECTO}/{registry['v1']}",
        "run_id": modelo["run_id"],
        "origen": "registry",
    }
    assert modelo["run_id"]


def test_carga_por_version_explicita(entorno, registry):
    entorno.setenv("MODELO_VERSION", registry["v2"])
    with TestClient(app) as c:
        modelo = c.get("/health").json()["modelo"]
    assert modelo["version"] == registry["v2"]
    assert modelo["alias"] is None


def test_version_no_numerica_se_rechaza(entorno):
    entorno.setenv("MODELO_VERSION", "ultima")
    with TestClient(app) as c:
        respuesta = c.get("/health")
    assert respuesta.status_code == 503
    assert "MODELO_VERSION debe ser un número" in respuesta.json()["detalle"]


def test_modelo_inexistente_responde_503(entorno):
    entorno.setenv("MODELO_NOMBRE", "no-existe")
    with TestClient(app) as c:
        salud = c.get("/health")
        prediccion = c.post("/predict/demanda", json=solicitud())
    assert salud.status_code == 503
    assert salud.json()["estado"] == "sin_modelo"
    assert prediccion.status_code == 503


def test_firma_distinta_a_p1_se_rechaza(entorno, registry):
    X, y = registry["X"].drop(columns="mes"), registry["X"]["salidas_rezago_24"]
    version = registrar(X, y, nombre="modelo-otra-firma")
    with pytest.raises(ErrorModelo, match="no coincide con FEATURES_P1"):
        cargar(Ajustes(nombre="modelo-otra-firma", version=version))


def test_servicio_reintenta_la_carga():
    intentos = []

    def cargador(_ajustes):
        intentos.append(1)
        if len(intentos) == 1:
            raise ConnectionError("servidor de MLflow apagado")
        return "modelo"

    servicio = Servicio(Ajustes(), cargador=cargador)
    with pytest.raises(ErrorModelo, match="servidor de MLflow apagado"):
        servicio.asegurar()
    assert servicio.asegurar() == "modelo"
    assert servicio.error is None


def test_descarga_y_carga_local_sin_servidor(entorno, registry, tmp_path):
    destino = tmp_path / "modelo_p1"
    resultado = CliRunner().invoke(descargar.app, ["--destino", str(destino)])
    assert resultado.exit_code == 0, resultado.output

    entorno.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # sin servidor
    entorno.setenv("MODELO_RUTA", str(destino))
    with TestClient(app) as c:
        modelo = c.get("/health").json()["modelo"]
        assert c.post("/predict/demanda", json=solicitud()).status_code == 200
    assert modelo["origen"] == "local"
    assert (modelo["nombre"], modelo["version"], modelo["alias"]) == (
        NOMBRE_POR_DEFECTO,
        registry["v1"],
        "champion",
    )


def test_ruta_local_sin_modelo_se_rechaza(tmp_path):
    with pytest.raises(ErrorModelo, match="falta MLmodel"):
        cargar(Ajustes(ruta_local=str(tmp_path)))


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def test_prediccion_coincide_con_el_modelo_registrado(cliente, registry):
    instancias = [
        instancia(estacion_id=1, periodo="2026-09-14T08:00:00"),
        instancia(estacion_id=2, periodo="2026-09-19T23:00:00", salidas_rezago_24=0),
        instancia(estacion_id=3, periodo="2026-12-01T00:00:00", llegadas_media_168=7.25),
    ]
    respuesta = cliente.post("/predict/demanda", json=solicitud(*instancias))
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()

    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_POR_DEFECTO}/{registry['v1']}")
    X = construir_variables([InstanciaDemanda(**i) for i in instancias])
    esperado = np.clip(modelo.predict(X), 0, None)
    obtenido = [p["salidas_esperadas"] for p in cuerpo["predicciones"]]
    assert obtenido == pytest.approx(esperado, abs=1e-4)
    assert [p["estacion_id"] for p in cuerpo["predicciones"]] == [1, 2, 3]
    assert cuerpo["modelo"]["version"] == registry["v1"]


def test_variables_de_calendario_iguales_al_entrenamiento():
    i = InstanciaDemanda(**instancia(periodo="2026-09-19T07:00:00"))  # sábado
    X = construir_variables([i])
    assert list(X.columns) == list(FEATURES_P1)
    fila = X.iloc[0]
    assert (fila.hora_del_dia, fila.dia_semana, fila.es_fin_de_semana, fila.mes) == (7, 5, 1, 9)
    assert fila.hora_sin == pytest.approx(math.sin(2 * math.pi * 7 / 24))
    assert fila.dia_cos == pytest.approx(math.cos(2 * math.pi * 5 / 7))
    assert fila.salidas_rezago_24 == EJEMPLO_INSTANCIA["salidas_rezago_24"]


def test_periodo_con_zona_horaria_se_convierte_a_hora_local():
    i = InstanciaDemanda(**instancia(periodo="2026-09-15T14:00:00Z"))
    assert i.periodo == pd.Timestamp("2026-09-15T08:00:00").to_pydatetime()
    assert construir_variables([i]).iloc[0].hora_del_dia == 8


class ModeloFijo:
    def __init__(self, valores):
        self.valores = valores

    def predict(self, X):
        return np.asarray(self.valores)


INFO = InfoModelo(
    nombre="x", version="1", alias=None, uri="models:/x/1", run_id=None, origen="registry"
)


def test_predicciones_negativas_se_recortan_en_cero():
    instancias = [InstanciaDemanda(**instancia(estacion_id=k)) for k in (1, 2)]
    salida = predecir(ModeloCargado(ModeloFijo([-0.3, 2.5]), INFO), instancias)
    assert [p.salidas_esperadas for p in salida] == [0.0, 2.5]


@pytest.mark.parametrize("valores", [[np.nan], [np.inf], [1.0, 2.0]])
def test_salida_invalida_del_modelo_es_error(valores):
    with pytest.raises(ErrorModelo):
        predecir(ModeloCargado(ModeloFijo(valores), INFO), [InstanciaDemanda(**instancia())])


# ---------------------------------------------------------------------------
# Validación de entradas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cambios", "mensaje"),
    [
        ({"estacion_id": 0}, "greater than or equal to 1"),
        ({"salidas_rezago_24": -1}, "greater than or equal to 0"),
        ({"llegadas_media_24": 501}, "less than or equal to 500"),
        ({"salidas_rezago_168": 2.5}, "valid integer"),
        ({"periodo": "2026-09-15T08:30:00"}, "hora exacta"),
        ({"periodo": "mañana"}, "datetime"),
        ({"salidas_media_24": "alto"}, "valid number"),
        ({"columna_extra": 1}, "Extra inputs are not permitted"),
    ],
)
def test_entradas_invalidas_responden_422(cliente, cambios, mensaje):
    respuesta = cliente.post("/predict/demanda", json=solicitud(instancia(**cambios)))
    assert respuesta.status_code == 422
    assert mensaje in str(respuesta.json()["detail"])


def test_falta_una_variable(cliente):
    datos = instancia()
    del datos["llegadas_media_168"]
    respuesta = cliente.post("/predict/demanda", json=solicitud(datos))
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"][-1] == "llegadas_media_168"


def test_valores_no_finitos_se_rechazan(cliente):
    # json.dumps escribe NaN (JSON no estándar), como lo enviaría un cliente descuidado.
    cuerpo = json.dumps(solicitud(instancia(salidas_media_24=float("nan"))))
    respuesta = cliente.post(
        "/predict/demanda", content=cuerpo, headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["input"] == "nan"


@pytest.mark.parametrize("n", [0, MAX_INSTANCIAS + 1])
def test_lote_vacio_o_demasiado_grande(cliente, n):
    instancias = [instancia(estacion_id=k + 1) for k in range(n)]
    assert cliente.post("/predict/demanda", json={"instancias": instancias}).status_code == 422


def test_instancias_duplicadas(cliente):
    respuesta = cliente.post("/predict/demanda", json=solicitud(instancia(), instancia()))
    assert respuesta.status_code == 422
    assert "repetidas" in str(respuesta.json()["detail"])


def test_documentacion_incluye_el_ejemplo(cliente):
    assert cliente.get("/docs").status_code == 200
    esquema = cliente.get("/openapi.json").json()
    ejemplo = esquema["components"]["schemas"]["SolicitudDemanda"]["examples"][0]
    assert ejemplo["instancias"][0] == EJEMPLO_INSTANCIA
    assert cliente.post("/predict/demanda", json=ejemplo).status_code == 200
