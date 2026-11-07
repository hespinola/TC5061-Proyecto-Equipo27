"""Aplicación FastAPI: salud del servicio y predicción de demanda (P1).

    uvicorn mibici.api.app:app --host 0.0.0.0 --port 8000

La documentación interactiva queda en ``/docs``.
"""

from contextlib import asynccontextmanager
import math
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mibici.api.esquemas import RespuestaDemanda, Salud, SolicitudDemanda
from mibici.api.modelo import Ajustes, ErrorModelo, Servicio, predecir


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    servicio = Servicio(Ajustes.desde_entorno())
    try:
        servicio.asegurar()
    except ErrorModelo:
        # El servicio arranca de todos modos: /health reporta el error y la carga se
        # reintenta en cada petición, por si el servidor de MLflow tarda en levantar.
        pass
    app.state.servicio = servicio
    yield


app = FastAPI(
    title="MiBici · demanda por estación y hora (P1)",
    description=(
        "Predice cuántos viajes iniciarán en una estación de MiBici en una hora dada, "
        "con el modelo `mibici-equipo27-p1-demanda` del Model Registry de MLflow."
    ),
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def _json_seguro(valor):
    """JSON no admite NaN ni infinito; se muestran como texto en el detalle del error."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, dict):
        return {k: _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [_json_seguro(v) for v in valor]
    return valor


@app.exception_handler(RequestValidationError)
async def error_de_validacion(request: Request, exc: RequestValidationError):
    # Mismo formato que FastAPI, pero sin fallar si la entrada traía NaN o infinito.
    detalle = _json_seguro(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": detalle})


def obtener_servicio(request: Request) -> Servicio:
    return request.app.state.servicio


ServicioDep = Annotated[Servicio, Depends(obtener_servicio)]


@app.get("/", include_in_schema=False)
def raiz():
    return {"servicio": app.title, "documentacion": "/docs", "salud": "/health"}


@app.get(
    "/health",
    response_model=Salud,
    responses={503: {"model": Salud, "description": "El modelo no está cargado."}},
)
def salud(servicio: ServicioDep):
    try:
        cargado = servicio.asegurar()
    except ErrorModelo:
        cuerpo = Salud(estado="sin_modelo", detalle=servicio.error)
        return JSONResponse(status_code=503, content=cuerpo.model_dump(mode="json"))
    return Salud(estado="ok", modelo=cargado.info)


@app.post(
    "/predict/demanda",
    response_model=RespuestaDemanda,
    responses={
        422: {"description": "La entrada no cumple el esquema."},
        503: {"description": "El modelo no está disponible."},
    },
)
def predecir_demanda(solicitud: SolicitudDemanda, servicio: ServicioDep) -> RespuestaDemanda:
    try:
        cargado = servicio.asegurar()
    except ErrorModelo as e:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible. {e}") from e
    try:
        predicciones = predecir(cargado, solicitud.instancias)
    except ErrorModelo as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return RespuestaDemanda(modelo=cargado.info, predicciones=predicciones)
