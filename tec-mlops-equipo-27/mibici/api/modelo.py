"""Carga del modelo de P1 desde el Model Registry de MLflow y predicción.

Dos formas de obtener el modelo, elegidas con variables de entorno:

* **Registry** (por defecto): ``MLFLOW_TRACKING_URI`` apunta al servidor de MLflow y
  el modelo se identifica por ``MODELO_NOMBRE`` y ``MODELO_VERSION`` o, si no se da
  versión, por ``MODELO_ALIAS`` (``champion``). El alias se resuelve primero a su
  número de versión y se carga esa versión exacta, para que la versión reportada
  por la API sea la que realmente responde.
* **Local**: ``MODELO_RUTA`` apunta a una copia descargada con
  ``python -m mibici.api.descargar``. No requiere servidor de MLflow, lo que permite
  empaquetar el modelo dentro de la imagen.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from mibici import features as f
from mibici.api.esquemas import InfoModelo, InstanciaDemanda, PrediccionDemanda
from mibici.config import FEATURES_P1

NOMBRE_POR_DEFECTO = "mibici-equipo27-p1-demanda"
ALIAS_POR_DEFECTO = "champion"
ARCHIVO_INFO = "modelo_info.json"


class ErrorModelo(RuntimeError):
    """El modelo no se pudo cargar o no cumple el contrato de P1."""


@dataclass(frozen=True)
class Ajustes:
    nombre: str = NOMBRE_POR_DEFECTO
    version: str | None = None
    alias: str = ALIAS_POR_DEFECTO
    ruta_local: str | None = None

    @classmethod
    def desde_entorno(cls) -> "Ajustes":
        def leer(variable):
            valor = os.getenv(variable, "").strip()
            return valor or None

        return cls(
            nombre=leer("MODELO_NOMBRE") or NOMBRE_POR_DEFECTO,
            version=leer("MODELO_VERSION"),
            alias=leer("MODELO_ALIAS") or ALIAS_POR_DEFECTO,
            ruta_local=leer("MODELO_RUTA"),
        )


@dataclass(frozen=True)
class ModeloCargado:
    modelo: object
    info: InfoModelo


def resolver_version(ajustes: Ajustes) -> tuple[str, str | None, str | None]:
    """Devuelve (versión, alias usado, run_id) consultando el Model Registry."""
    cliente = MlflowClient()
    if ajustes.version:
        if not ajustes.version.isdigit():
            raise ErrorModelo(f"MODELO_VERSION debe ser un número, no {ajustes.version!r}.")
        mv = cliente.get_model_version(ajustes.nombre, ajustes.version)
        alias = None
    else:
        mv = cliente.get_model_version_by_alias(ajustes.nombre, ajustes.alias)
        alias = ajustes.alias
    # Según el backend, MLflow devuelve la versión como texto (REST) o como entero (SQL).
    return str(mv.version), alias, mv.run_id


def validar_firma(modelo) -> None:
    """Exige que la firma registrada tenga exactamente las variables de P1, en orden."""
    esquema = modelo.metadata.get_input_schema()
    if esquema is None:
        raise ErrorModelo("El modelo no tiene firma de entrada registrada.")
    columnas = esquema.input_names()
    if columnas != list(FEATURES_P1):
        raise ErrorModelo(
            "La firma del modelo no coincide con FEATURES_P1. "
            f"Esperadas {list(FEATURES_P1)}, recibidas {columnas}."
        )


def cargar(ajustes: Ajustes) -> ModeloCargado:
    if ajustes.ruta_local:
        ruta = Path(ajustes.ruta_local)
        if not (ruta / "MLmodel").is_file():
            raise ErrorModelo(f"{ruta} no contiene un modelo de MLflow (falta MLmodel).")
        modelo = mlflow.pyfunc.load_model(str(ruta))
        datos = (
            json.loads((ruta / ARCHIVO_INFO).read_text())
            if (ruta / ARCHIVO_INFO).is_file()
            else {}
        )
        info = InfoModelo(
            nombre=datos.get("nombre"),
            version=datos.get("version"),
            alias=datos.get("alias"),
            uri=str(ruta),
            run_id=datos.get("run_id") or modelo.metadata.run_id,
            origen="local",
        )
    else:
        version, alias, run_id = resolver_version(ajustes)
        uri = f"models:/{ajustes.nombre}/{version}"
        modelo = mlflow.pyfunc.load_model(uri)
        info = InfoModelo(
            nombre=ajustes.nombre,
            version=version,
            alias=alias,
            uri=uri,
            run_id=run_id,
            origen="registry",
        )
    validar_firma(modelo)
    logger.info(f"Modelo cargado: {info.model_dump()}")
    return ModeloCargado(modelo, info)


def construir_variables(instancias: list[InstanciaDemanda]) -> pd.DataFrame:
    """Arma la matriz de entrada con las columnas y el orden de FEATURES_P1."""
    datos = pd.DataFrame([i.model_dump() for i in instancias])
    datos[f.COL_PERIODO] = pd.to_datetime(datos[f.COL_PERIODO])
    datos = f.agregar_variables_calendario(datos)
    return datos[list(FEATURES_P1)].astype(float)


def predecir(
    cargado: ModeloCargado, instancias: list[InstanciaDemanda]
) -> list[PrediccionDemanda]:
    salida = np.asarray(cargado.modelo.predict(construir_variables(instancias)), dtype=float)
    salida = salida.reshape(-1)
    if len(salida) != len(instancias):
        raise ErrorModelo(
            f"El modelo devolvió {len(salida)} valores para {len(instancias)} filas."
        )
    if not np.isfinite(salida).all():
        raise ErrorModelo("El modelo devolvió valores no finitos.")
    # El regresor puede dar valores ligeramente negativos en horas sin demanda; un
    # conteo de viajes no puede serlo, así que se recorta en 0.
    salida = np.clip(salida, 0, None)
    return [
        PrediccionDemanda(
            estacion_id=i.estacion_id, periodo=i.periodo, salidas_esperadas=round(float(y), 4)
        )
        for i, y in zip(instancias, salida, strict=True)
    ]


class Servicio:
    """Mantiene el modelo cargado y reintenta la carga si el registry no respondía."""

    def __init__(self, ajustes: Ajustes, cargador=cargar):
        self.ajustes = ajustes
        self._cargador = cargador
        self._lock = threading.Lock()
        self.cargado: ModeloCargado | None = None
        self.error: str | None = None

    def asegurar(self) -> ModeloCargado:
        if self.cargado is not None:
            return self.cargado
        with self._lock:
            if self.cargado is None:
                try:
                    self.cargado = self._cargador(self.ajustes)
                    self.error = None
                except Exception as e:  # el detalle se expone en /health
                    self.error = f"{type(e).__name__}: {e}"
                    logger.error(f"No se pudo cargar el modelo: {self.error}")
                    raise ErrorModelo(self.error) from e
        return self.cargado
