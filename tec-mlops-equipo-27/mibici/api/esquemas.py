"""Contratos de entrada y salida de la API (Pydantic).

El cliente envía la estación, la hora a predecir y el historial de demanda que ya
conoce. Las variables de calendario (hora, día, mes y sus codificaciones seno y
coseno) no se piden: la API las calcula con ``features.agregar_variables_calendario``,
la misma función del entrenamiento, para evitar diferencias entre cómo se entrenó
y cómo se sirve el modelo.
"""

from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

ZONA_HORARIA = ZoneInfo("America/Mexico_City")  # Guadalajara usa la hora del centro.
MAX_INSTANCIAS = 1000
# Cota de sanidad por estación y hora. Ninguna estación de MiBici se acerca a este
# volumen; un valor mayor indica un error del cliente (por ejemplo, totales diarios).
MAX_VIAJES_HORA = 500


def _hora_local_exacta(valor: datetime) -> datetime:
    if valor.tzinfo is not None:
        valor = valor.astimezone(ZONA_HORARIA).replace(tzinfo=None)
    if (valor.minute, valor.second, valor.microsecond) != (0, 0, 0):
        raise ValueError("periodo debe ser una hora exacta (minutos y segundos en 0).")
    return valor


Conteo = Annotated[int, Field(ge=0, le=MAX_VIAJES_HORA)]
Promedio = Annotated[float, Field(ge=0, le=MAX_VIAJES_HORA, allow_inf_nan=False)]


class InstanciaDemanda(BaseModel):
    """Una estación en una hora, con la demanda observada antes de esa hora."""

    model_config = ConfigDict(extra="forbid")

    estacion_id: int = Field(ge=1, description="ID de la estación en la nomenclatura de MiBici.")
    periodo: Annotated[datetime, AfterValidator(_hora_local_exacta)] = Field(
        description=(
            "Hora a predecir, en hora local de Guadalajara. Si trae zona horaria, "
            "se convierte a America/Mexico_City."
        )
    )
    salidas_rezago_24: Conteo = Field(description="Salidas en la misma hora del día anterior.")
    salidas_rezago_168: Conteo = Field(
        description="Salidas en la misma hora de la semana anterior."
    )
    salidas_media_24: Promedio = Field(
        description="Promedio de salidas por hora en las últimas 24 h."
    )
    salidas_media_168: Promedio = Field(
        description="Promedio de salidas por hora en las últimas 168 h."
    )
    llegadas_rezago_24: Conteo = Field(description="Llegadas en la misma hora del día anterior.")
    llegadas_rezago_168: Conteo = Field(
        description="Llegadas en la misma hora de la semana anterior."
    )
    llegadas_media_24: Promedio = Field(
        description="Promedio de llegadas por hora en las últimas 24 h."
    )
    llegadas_media_168: Promedio = Field(
        description="Promedio de llegadas por hora en las últimas 168 h."
    )


EJEMPLO_INSTANCIA = {
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


class SolicitudDemanda(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"instancias": [EJEMPLO_INSTANCIA]}]},
    )

    instancias: list[InstanciaDemanda] = Field(min_length=1, max_length=MAX_INSTANCIAS)

    @model_validator(mode="after")
    def _sin_duplicados(self):
        claves = [(i.estacion_id, i.periodo) for i in self.instancias]
        if len(claves) != len(set(claves)):
            raise ValueError("Hay instancias repetidas (misma estacion_id y periodo).")
        return self


class InfoModelo(BaseModel):
    nombre: str | None = Field(description="Nombre en el Model Registry.")
    version: str | None = Field(description="Versión del Model Registry que se cargó.")
    alias: str | None = Field(description="Alias usado para resolver la versión, si aplica.")
    uri: str = Field(description="URI con la que se cargó el modelo.")
    run_id: str | None = Field(description="Corrida de MLflow que entrenó el modelo.")
    origen: str = Field(description="'registry' (servidor de MLflow) o 'local' (empaquetado).")


class PrediccionDemanda(BaseModel):
    estacion_id: int
    periodo: datetime
    salidas_esperadas: float = Field(
        ge=0, allow_inf_nan=False, description="Viajes que se espera inicien en esa hora."
    )


class RespuestaDemanda(BaseModel):
    modelo: InfoModelo
    predicciones: list[PrediccionDemanda]


class Salud(BaseModel):
    estado: str = Field(description="'ok' si el modelo está cargado; 'sin_modelo' si no.")
    modelo: InfoModelo | None = None
    detalle: str | None = None
