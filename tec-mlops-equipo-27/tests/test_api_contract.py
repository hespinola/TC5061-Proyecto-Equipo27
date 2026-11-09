"""Contrato HTTP de P1, sin entrenar ni contactar el Model Registry.

Ejecutar: python -m pytest tests/test_api_contract.py -v
Se conserva el ciclo de vida, las rutas y la validación reales; solo se sustituye
el cargador del servicio. Las expectativas JSON son independientes de los esquemas.
"""

from copy import deepcopy
import importlib
import math

from fastapi.testclient import TestClient
import numpy as np
import pytest

from mibici.api.esquemas import InfoModelo
from mibici.api.modelo import ErrorModelo, ModeloCargado, Servicio

api = importlib.import_module("mibici.api.app")

INSTANCIA = {
    "estacion_id": 12,
    "periodo": "2026-09-15T08:00:00",
    "salidas_rezago_24": 6,
    "salidas_rezago_168": 5,
    "salidas_media_24": 1.8,
    "salidas_media_168": 1.6,
    "llegadas_rezago_24": 3,
    "llegadas_rezago_168": 4,
    "llegadas_media_24": 1.7,
    "llegadas_media_168": 1.5,
}
INFO = {
    "nombre": "mibici-equipo27-p1-demanda",
    "version": "7",
    "alias": "champion",
    "uri": "models:/mibici-equipo27-p1-demanda/7",
    "run_id": "corrida-de-contrato",
    "origen": "registry",
}


class ModeloControlado:
    def __init__(self):
        self.salida = None
        self.llamadas = 0

    def predict(self, variables):
        self.llamadas += 1
        if self.salida is not None:
            return self.salida
        return variables["salidas_rezago_24"].to_numpy() + 0.25


@pytest.fixture
def cliente(monkeypatch):
    modelo = ModeloControlado()
    cargado = ModeloCargado(modelo, InfoModelo(**INFO))
    servicio = Servicio(None, cargador=lambda _: cargado)
    # Evita incluso el intento de conexión durante el arranque de TestClient.
    monkeypatch.setattr(api, "Servicio", lambda _: servicio)
    with TestClient(api.app) as client:
        yield client, modelo, servicio


def json_de(respuesta, codigo):
    assert respuesta.status_code == codigo, respuesta.text
    assert respuesta.headers["content-type"].split(";")[0] == "application/json"
    return respuesta.json()


def test_contrato_health_disponible(cliente):
    client, _, _ = cliente
    assert json_de(client.get("/health"), 200) == {"estado": "ok", "modelo": INFO, "detalle": None}


def test_contrato_prediccion_preserva_orden_identidad_y_tipos(cliente):
    client, _, _ = cliente
    instancias = [
        {**INSTANCIA, "estacion_id": 30, "salidas_rezago_24": 9},
        {**INSTANCIA, "estacion_id": 2, "periodo": "2026-09-16T14:00:00Z"},
    ]
    cuerpo = json_de(client.post("/predict/demanda", json={"instancias": instancias}), 200)
    assert cuerpo == {
        "modelo": INFO,
        "predicciones": [
            {"estacion_id": 30, "periodo": "2026-09-15T08:00:00", "salidas_esperadas": 9.25},
            {"estacion_id": 2, "periodo": "2026-09-16T08:00:00", "salidas_esperadas": 6.25},
        ],
    }
    for prediccion in cuerpo["predicciones"]:
        assert type(prediccion["estacion_id"]) is int
        assert type(prediccion["salidas_esperadas"]) is float
        assert math.isfinite(prediccion["salidas_esperadas"])


@pytest.mark.parametrize("cantidad", [1, 1000])
@pytest.mark.parametrize("valor", [0, 500])
def test_contrato_acepta_limites_inclusivos(cliente, cantidad, valor):
    client, _, _ = cliente
    datos = {k: valor if k not in ("estacion_id", "periodo") else v for k, v in INSTANCIA.items()}
    instancias = [{**datos, "estacion_id": k + 1} for k in range(cantidad)]
    cuerpo = json_de(client.post("/predict/demanda", json={"instancias": instancias}), 200)
    assert len(cuerpo["predicciones"]) == cantidad
    assert [p["estacion_id"] for p in cuerpo["predicciones"]] == list(range(1, cantidad + 1))
    assert all(p["salidas_esperadas"] == valor + 0.25 for p in cuerpo["predicciones"])


@pytest.mark.parametrize(
    "peticion,ubicacion,tipo",
    [
        ({}, ["body"], "missing"),
        ({"content": "null"}, ["body"], "model_attributes_type"),
        ({"json": {}}, ["body", "instancias"], "missing"),
        ({"json": []}, ["body"], "model_attributes_type"),
        ({"json": {"instancias": None}}, ["body", "instancias"], "list_type"),
        ({"json": {"instancias": [None]}}, ["body", "instancias", 0], "model_attributes_type"),
        ({"json": {"instancias": [INSTANCIA], "extra": 1}}, ["body", "extra"], "extra_forbidden"),
        (
            {"json": {"instancias": [{**INSTANCIA, "periodo": None}]}},
            ["body", "instancias", 0, "periodo"],
            "datetime_type",
        ),
    ],
)
def test_contrato_422_estructura_y_ubicacion(cliente, peticion, ubicacion, tipo):
    client, modelo, _ = cliente
    cuerpo = json_de(
        client.post(
            "/predict/demanda",
            headers={"Content-Type": "application/json"},
            **deepcopy(peticion),
        ),
        422,
    )
    assert set(cuerpo) == {"detail"}
    assert isinstance(cuerpo["detail"], list) and cuerpo["detail"]
    error = cuerpo["detail"][0]
    assert error["loc"] == ubicacion
    assert error["type"] == tipo
    assert isinstance(error["msg"], str) and error["msg"]
    assert modelo.llamadas == 0


def test_contrato_rechaza_json_malformado(cliente):
    client, modelo, _ = cliente
    cuerpo = json_de(
        client.post(
            "/predict/demanda",
            content='{"instancias":',
            headers={"Content-Type": "application/json"},
        ),
        422,
    )
    assert cuerpo["detail"][0]["type"] == "json_invalid"
    assert cuerpo["detail"][0]["loc"][0] == "body"
    assert modelo.llamadas == 0


def test_contrato_rechaza_duplicados_tras_normalizar_zona(cliente):
    client, modelo, _ = cliente
    cuerpo = json_de(
        client.post(
            "/predict/demanda",
            json={"instancias": [INSTANCIA, {**INSTANCIA, "periodo": "2026-09-15T14:00:00Z"}]},
        ),
        422,
    )
    assert cuerpo["detail"][0]["loc"] == ["body"]
    assert cuerpo["detail"][0]["type"] == "value_error"
    assert modelo.llamadas == 0


def test_contrato_503_y_recuperacion(cliente, monkeypatch):
    client, modelo, servicio = cliente
    cargado = servicio.cargado

    def no_disponible(_):
        raise ErrorModelo("Registry temporalmente no disponible")

    servicio.cargado = None
    monkeypatch.setattr(servicio, "_cargador", no_disponible)
    salud = json_de(client.get("/health"), 503)
    assert set(salud) == {"estado", "modelo", "detalle"}
    assert salud["estado"] == "sin_modelo" and salud["modelo"] is None
    assert isinstance(salud["detalle"], str) and salud["detalle"]
    fallo = json_de(client.post("/predict/demanda", json={"instancias": [INSTANCIA]}), 503)
    assert set(fallo) == {"detail"}
    assert isinstance(fallo["detail"], str) and fallo["detail"]
    assert modelo.llamadas == 0

    monkeypatch.setattr(servicio, "_cargador", lambda _: cargado)
    assert json_de(client.get("/health"), 200) == {"estado": "ok", "modelo": INFO, "detalle": None}
    json_de(client.post("/predict/demanda", json={"instancias": [INSTANCIA]}), 200)


@pytest.mark.parametrize("salida", [[np.nan], [np.inf], [-np.inf], [], [1.0, 2.0]])
def test_contrato_500_no_publica_predicciones_invalidas(cliente, salida):
    client, modelo, _ = cliente
    modelo.salida = salida
    cuerpo = json_de(client.post("/predict/demanda", json={"instancias": [INSTANCIA]}), 500)
    assert set(cuerpo) == {"detail"}
    assert isinstance(cuerpo["detail"], str) and cuerpo["detail"]


def test_contrato_openapi_rutas_y_esquemas(cliente):
    client, _, _ = cliente
    esquema = json_de(client.get("/openapi.json"), 200)
    salud = esquema["paths"]["/health"]
    demanda = esquema["paths"]["/predict/demanda"]
    assert set(salud) == {"get"}
    assert set(demanda) == {"post"}
    assert {"200", "503"} <= salud["get"]["responses"].keys()
    assert {"200", "422", "503"} <= demanda["post"]["responses"].keys()
    entrada = demanda["post"]["requestBody"]
    assert entrada["required"] is True
    assert entrada["content"]["application/json"]["schema"]["$ref"].endswith("/SolicitudDemanda")
    salida = demanda["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert salida["$ref"].endswith("/RespuestaDemanda")
    modelos = esquema["components"]["schemas"]
    assert set(modelos["InstanciaDemanda"]["required"]) == set(INSTANCIA)
    assert modelos["InstanciaDemanda"]["additionalProperties"] is False
    solicitud = modelos["SolicitudDemanda"]
    assert solicitud["required"] == ["instancias"]
    assert solicitud["additionalProperties"] is False
    assert solicitud["properties"]["instancias"]["minItems"] == 1
    assert solicitud["properties"]["instancias"]["maxItems"] == 1000
    assert set(modelos["RespuestaDemanda"]["required"]) == {"modelo", "predicciones"}
    assert set(modelos["PrediccionDemanda"]["required"]) == {
        "estacion_id",
        "periodo",
        "salidas_esperadas",
    }
