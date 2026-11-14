"""Preparación de datos para comparar un mes nuevo contra el periodo congelado.

Las llamadas a Evidently viven en el notebook ``2.0-equipo27-drift-evidently-mibici``;
aquí está lo reutilizable y comprobable sin Evidently:

* descarga y limpieza del mes nuevo con las mismas funciones de ``mibici.dataset``
  (mismos nombres de columna, codificación, tipos y reglas de descarte);
* construcción de variables comparables (duración, hora, día, estación);
* resumen de calidad por variable y lectura de los resultados de Evidently.

No modifica ``data_manifest.json``: el periodo congelado del proyecto sigue siendo
``PERIODO_INICIO``..``PERIODO_FIN``. El mes nuevo se descarga a ``data/raw`` y no se
mezcla con ``data/processed``.

Solo se usan los campos operativos que ya conserva ``dataset.py``. Los campos
personales del CSV crudo (usuario, género, año de nacimiento) no se leen.
"""

import json
from pathlib import Path
import re

from loguru import logger
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mibici.config import (
    DATA_MANIFEST_PATH,
    MES_NUEVO,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    URLS_VIAJES,
)
from mibici.dataset import descargar, detectar_codificacion, normalizar_viajes

VARIABLES_NUMERICAS = ("duracion_min", "capacidad_origen", "latitud_origen", "longitud_origen")
# Discretas: se comparan como categorías (distancia de Jensen-Shannon en Evidently).
VARIABLES_CATEGORICAS = (
    "hora_inicio",
    "dia_semana",
    "es_fin_de_semana",
    "origen_id",
    "destino_id",
)
VARIABLES = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS

# Cotas heurísticas solo para describir calidad; no filtran datos.
DURACION_CORTA_MIN = 1.0
DURACION_LARGA_MIN = 240.0


def _ruta_mes(mes: str, raw_dir: Path) -> Path:
    return raw_dir / f"datos_abiertos_{mes.replace('-', '_')}.csv"


def descargar_mes(mes: str = MES_NUEVO, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Descarga atómica del CSV del mes (reutiliza ``dataset.descargar``) sin hash fijado."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    destino = _ruta_mes(mes, raw_dir)
    with requests.Session() as session:
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
                )
            ),
        )
        descargar(URLS_VIAJES[mes], destino, None, session)
    return destino


def cargar_mes(mes: str = MES_NUEVO, raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """Lee y normaliza el CSV crudo de un mes con las reglas de ``dataset.py``.

    Devuelve los viajes válidos y los contadores de descarte, con el mismo significado
    que en el manifiesto (``originales``, ``invalidos``, ``fuera_periodo``,
    ``duplicados``, ``validos``). A diferencia de ``ingestar``, un ``viaje_id`` repetido
    se conserva una sola vez sin abortar: aquí se mide la calidad, no se congela una versión.
    """
    path = _ruta_mes(mes, raw_dir)
    inicio = pd.Timestamp(f"{mes}-01")
    fin = inicio + pd.offsets.MonthBegin(1)
    crudo = pd.read_csv(path, encoding=detectar_codificacion(path))
    viajes, stats = normalizar_viajes(crudo, str(inicio), str(fin))
    antes = len(viajes)
    viajes = viajes.drop_duplicates("viaje_id", keep="first").reset_index(drop=True)
    stats["duplicados"] = antes - len(viajes)
    stats["validos"] = len(viajes)
    stats["archivo"] = path.name
    logger.info(f"{mes}: {stats['validos']:,} viajes válidos de {stats['originales']:,}")
    return viajes, stats


def cargar_referencia(
    meses: list[str] | None = None, processed_dir: Path = PROCESSED_DATA_DIR
) -> pd.DataFrame:
    """Viajes ya normalizados del periodo congelado (``make data`` / ``mlflow run -e ingest``)."""
    path = processed_dir / "viajes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecutar antes la ingesta: mlflow run . -e ingest "
            "(o python -m mibici.dataset desde tec-mlops-equipo-27)."
        )
    viajes = pd.read_parquet(path)
    if meses:
        periodo = viajes["inicio_del_viaje"].dt.strftime("%Y-%m")
        viajes = viajes.loc[periodo.isin(meses)].reset_index(drop=True)
        if viajes.empty:
            raise ValueError(f"La referencia no contiene viajes de {meses}.")
    return viajes


def cargar_nomenclatura(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / "nomenclatura.parquet")


def construir_variables(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Variables comparables entre periodos; ``viaje_id`` y las fechas absolutas se excluyen.

    ``viaje_id`` crece con el tiempo y las fechas absolutas nunca se traslapan entre un
    periodo y el siguiente, así que "driftearían" siempre sin informar nada. Se usan en su
    lugar la duración y los componentes cíclicos del inicio del viaje.
    """
    inicio = viajes["inicio_del_viaje"]
    est = estaciones.set_index("id")
    origen, destino = viajes["origen_id"], viajes["destino_id"]
    return pd.DataFrame(
        {
            "duracion_min": (viajes["fin_del_viaje"] - inicio).dt.total_seconds() / 60,
            "hora_inicio": inicio.dt.hour.astype(str),
            "dia_semana": inicio.dt.dayofweek.astype(str),
            "es_fin_de_semana": (inicio.dt.dayofweek >= 5).astype(str),
            "origen_id": origen.astype(str),
            "destino_id": destino.astype(str),
            "capacidad_origen": origen.map(est["dpcapacity"]).astype("float64"),
            "latitud_origen": origen.map(est["latitude"]).astype("float64"),
            "longitud_origen": origen.map(est["longitude"]).astype("float64"),
        }
    )[list(VARIABLES)]


def muestrear(df: pd.DataFrame, n: int | None, seed: int = SEED) -> pd.DataFrame:
    """Muestra reproducible; Evidently calcula distancias sobre toda la muestra."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calidad
# ---------------------------------------------------------------------------
def descartes_referencia(manifest_path: Path = DATA_MANIFEST_PATH) -> pd.DataFrame:
    """Contadores de descarte por mes tomados del manifiesto versionado."""
    archivos = json.loads(manifest_path.read_text(encoding="utf-8"))["archivos"]
    columnas = ["originales", "invalidos", "fuera_periodo", "duplicados", "validos"]
    return pd.DataFrame({m: v for m, v in archivos.items() if m != "nomenclatura"}).T[columnas]


def tabla_descartes(referencia: pd.DataFrame, nuevo: dict, mes: str = MES_NUEVO) -> pd.DataFrame:
    """Une los descartes del periodo congelado con los del mes nuevo y agrega porcentajes."""
    cols = list(referencia.columns)
    tabla = pd.concat([referencia, pd.DataFrame([{c: nuevo[c] for c in cols}], index=[mes])])
    tabla = tabla.astype("int64")
    for c in ("invalidos", "fuera_periodo", "duplicados"):
        tabla[f"{c}_pct"] = (100 * tabla[c] / tabla["originales"]).round(3)
    return tabla


def resumen_calidad(
    referencia: pd.DataFrame, actual: pd.DataFrame, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Chequeos de calidad lado a lado sobre las variables ya construidas (antes de muestrear)."""
    ids_validos = set(estaciones["id"].astype(str))
    ids_ref = set(referencia["origen_id"]) | set(referencia["destino_id"])
    ids_act = set(actual["origen_id"]) | set(actual["destino_id"])

    def medir(df: pd.DataFrame) -> dict:
        d = df["duracion_min"]
        return {
            "filas": len(df),
            "filas_duplicadas_pct": 100 * df.duplicated().mean(),
            "faltantes_totales_pct": 100 * df.isna().to_numpy().mean(),
            "duracion_negativa_o_cero_pct": 100 * (d <= 0).mean(),
            f"duracion_menor_{DURACION_CORTA_MIN:g}_min_pct": 100
            * (d < DURACION_CORTA_MIN).mean(),
            f"duracion_mayor_{DURACION_LARGA_MIN:g}_min_pct": 100
            * (d > DURACION_LARGA_MIN).mean(),
            "origen_sin_nomenclatura_pct": 100 * (~df["origen_id"].isin(ids_validos)).mean(),
            "destino_sin_nomenclatura_pct": 100 * (~df["destino_id"].isin(ids_validos)).mean(),
            "estaciones_distintas": len(set(df["origen_id"]) | set(df["destino_id"])),
        }

    tabla = pd.DataFrame({"referencia": medir(referencia), "actual": medir(actual)})
    tabla["diferencia"] = tabla["actual"] - tabla["referencia"]
    tabla.loc["estaciones_nuevas_en_actual"] = [np.nan, len(ids_act - ids_ref), np.nan]
    tabla.loc["estaciones_ausentes_en_actual"] = [np.nan, len(ids_ref - ids_act), np.nan]
    return tabla


def resumen_variables(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Faltantes, valores únicos y rango por variable, para ambos conjuntos."""
    filas = []
    for col in referencia.columns:
        fila = {"variable": col}
        for nombre, df in (("ref", referencia), ("act", actual)):
            fila[f"{nombre}_faltantes_pct"] = round(100 * df[col].isna().mean(), 4)
            fila[f"{nombre}_unicos"] = df[col].nunique()
            if col in VARIABLES_NUMERICAS:
                fila[f"{nombre}_min"] = df[col].min()
                fila[f"{nombre}_mediana"] = df[col].median()
                fila[f"{nombre}_max"] = df[col].max()
        filas.append(fila)
    return pd.DataFrame(filas).set_index("variable")


# ---------------------------------------------------------------------------
# Lectura de resultados de Evidently
# ---------------------------------------------------------------------------
def tabla_drift(resultado: dict) -> pd.DataFrame:
    """Drift por variable a partir de ``snapshot.dict()`` de un Report con DataDriftPreset.

    ``drift`` usa el veredicto de la prueba de Evidently (``FAIL`` = drift detectado),
    de modo que respeta la dirección de cada método (distancia vs. valor p).
    """
    estados = {}
    for prueba in resultado.get("tests", []):
        m = re.fullmatch(r"Value Drift for column (.)", prueba["name"])
        if m:
            estados[m.group(1)] = str(prueba["status"]).rsplit(".", 1)[-1] == "FAIL"
    filas = []
    for metrica in resultado["metrics"]:
        cfg = metrica["config"]
        if cfg["type"].endswith(":ValueDrift"):
            col = cfg["column"]
            filas.append(
                {
                    "variable": col,
                    "tipo": "numérica" if col in VARIABLES_NUMERICAS else "categórica",
                    "metodo": cfg["method"],
                    "umbral": cfg["threshold"],
                    "puntaje": float(metrica["value"]),
                    "drift": estados.get(col),
                }
            )
    return pd.DataFrame(filas).set_index("variable").sort_values("puntaje", ascending=False)