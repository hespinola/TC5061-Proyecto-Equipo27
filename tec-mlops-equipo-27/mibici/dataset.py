"""Ingesta por bloques, normalización y manifiesto verificable de MiBici.

Git conserva el manifiesto; los datos quedan fuera de Git. Un cambio de fuente
o transformación requiere ``--actualizar-version`` y revisión del manifiesto.
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
import typer
from urllib3.util.retry import Retry

from mibici.config import (
    CABECERAS_DESCARGA,
    CHUNK_SIZE,
    DATA_MANIFEST_PATH,
    EQUIPO,
    MESES,
    MLFLOW_TRACKING_URI,
    OBSERVACION_FIN,
    OBSERVACION_INICIO,
    PORTAL_DATOS_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TIMEOUT_DESCARGA,
    URL_NOMENCLATURA,
    URLS_VIAJES,
    VERSION_PREPROCESAMIENTO,
)

app = typer.Typer()
SCHEMA = pa.schema(
    [
        ("viaje_id", pa.int64()),
        ("inicio_del_viaje", pa.timestamp("us")),
        ("fin_del_viaje", pa.timestamp("us")),
        ("origen_id", pa.int64()),
        ("destino_id", pa.int64()),
    ]
)
ALIASES = {
    "viaje_id": "viaje_id",
    "id_viaje": "viaje_id",
    "inicio_del_viaje": "inicio_del_viaje",
    "inicio_viaje": "inicio_del_viaje",
    "fin_del_viaje": "fin_del_viaje",
    "fin_viaje": "fin_del_viaje",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for bloque in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def nombre_columna(nombre: str) -> str:
    texto = unicodedata.normalize("NFKD", str(nombre).strip().lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")


def detectar_codificacion(path: Path) -> str:
    """Prueba codificaciones determinísticamente, validando todo el archivo."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(encoding=encoding) as stream:
                while stream.read(1024 * 1024):
                    pass
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {path.name}")


def normalizar_viajes(df: pd.DataFrame, inicio: str, fin: str):
    """Retiene solo campos operativos y reporta filas inválidas/fuera del periodo."""
    df = df.rename(columns=lambda c: ALIASES.get(nombre_columna(c), nombre_columna(c)))
    if df.columns.duplicated().any():
        raise ValueError("Columnas duplicadas tras normalizar nombres.")
    faltantes = set(SCHEMA.names) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas operativas: {sorted(faltantes)}")
    out = df[SCHEMA.names].copy()
    for col in ("inicio_del_viaje", "fin_del_viaje"):
        # Hora civil local publicada; no convertir a UTC ni usar campos personales.
        out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            raise ValueError("Se esperan fechas locales sin zona horaria en los CSV de MiBici.")
    for col in ("viaje_id", "origen_id", "destino_id"):
        v = pd.to_numeric(out[col], errors="coerce")
        out[col] = v.where(np.isfinite(v) & (v > 0) & (v % 1 == 0))
    invalida = out.isna().any(axis=1)
    invalida |= out.fin_del_viaje < out.inicio_del_viaje
    fuera = ~out.inicio_del_viaje.between(
        pd.Timestamp(inicio), pd.Timestamp(fin), inclusive="left"
    )
    stats = {
        "originales": len(out),
        "invalidos": int(invalida.sum()),
        "fuera_periodo": int((~invalida & fuera).sum()),
    }
    out = out.loc[~invalida & ~fuera].copy()
    for col in ("viaje_id", "origen_id", "destino_id"):
        out[col] = out[col].astype("int64")
    return out, stats


def normalizar_estaciones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=nombre_columna)
    requeridas = ["id", "latitude", "longitude", "dpcapacity"]
    if df.columns.duplicated().any() or not set(requeridas) <= set(df.columns):
        raise ValueError(f"La nomenclatura requiere columnas únicas: {requeridas}.")
    out = df[[*requeridas, *(["name"] if "name" in df else [])]].copy()
    for c in requeridas:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if (
        out.id.isna().any()
        or (out.id <= 0).any()
        or (out.id % 1 != 0).any()
        or out.id.duplicated().any()
    ):
        raise ValueError("Nomenclatura con identificadores inválidos o duplicados.")
    out["id"] = out.id.astype("int64")
    out["dpcapacity"] = out.dpcapacity.where(np.isfinite(out.dpcapacity) & (out.dpcapacity > 0))
    out["latitude"] = out.latitude.where(out.latitude.between(-90, 90))
    out["longitude"] = out.longitude.where(out.longitude.between(-180, 180))
    return out.sort_values("id", ignore_index=True)


def solicitar_csv(url, session):
    """Solicita CSV con cabeceras explícitas y un único reintento tras HTTP 406.

    Si el servidor requiere una sesión iniciada desde su portal público, se visita
    esa página una vez y se reutilizan sus cookies. Un segundo 406 se propaga; no
    se agregan reintentos ilimitados ni se desactiva TLS.
    """
    kwargs = {"stream": True, "timeout": TIMEOUT_DESCARGA, "headers": CABECERAS_DESCARGA}
    response = session.get(url, **kwargs)
    if response.status_code == 406:
        response.close()
        logger.warning(
            "HTTP 406: se iniciará sesión en el portal público y se reintentará una vez."
        )
        cabeceras_portal = {**CABECERAS_DESCARGA, "Accept": "text/html,*/*;q=0.8"}
        with session.get(
            PORTAL_DATOS_URL, headers=cabeceras_portal, timeout=TIMEOUT_DESCARGA
        ) as portal:
            portal.raise_for_status()
        response = session.get(url, **kwargs)
    return response


def descargar(url, destino, hash_esperado, session, forzar=False):
    """Descarga atómica; rechaza una caché que no coincida con su hash fijado."""
    if forzar or not destino.exists():
        temporal = destino.with_suffix(destino.suffix + ".part")
        try:
            with solicitar_csv(url, session) as response:
                if response.status_code == 406:
                    raise requests.HTTPError(
                        f"MiBici mantiene HTTP 406 tras iniciar sesión en su portal: {url}. "
                        "Comprobar si el enlace abre desde un navegador en la misma red. "
                        "No se guardó la respuesta como datos.",
                        response=response,
                    )
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError(f"La fuente devolvió HTML en lugar de CSV: {url}")
                with temporal.open("wb") as stream:
                    primer_bloque = True
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        if primer_bloque and chunk.lstrip().lower().startswith(
                            (b"<!doctype html", b"<html")
                        ):
                            raise ValueError(f"La fuente devolvió HTML en lugar de CSV: {url}")
                        primer_bloque = False
                        stream.write(chunk)
                    if primer_bloque:
                        raise ValueError(f"La fuente devolvió un archivo vacío: {url}")
            temporal.replace(destino)
        finally:
            temporal.unlink(missing_ok=True)
    if hash_esperado and sha256_archivo(destino) != hash_esperado:
        raise ValueError(f"Hash distinto para {destino.name}; no coincide con la versión fijada.")


def version_manifest(payload: dict) -> str:
    contenido = {k: v for k, v in payload.items() if k != "data_version"}
    return hashlib.sha256(json.dumps(contenido, sort_keys=True).encode()).hexdigest()


def ingestar(
    raw_dir=RAW_DATA_DIR,
    processed_dir=PROCESSED_DATA_DIR,
    manifest_path=DATA_MANIFEST_PATH,
    meses=None,
    urls=None,
    nomenclatura_url=URL_NOMENCLATURA,
    inicio=OBSERVACION_INICIO,
    fin=OBSERVACION_FIN,
    actualizar_version=False,
):
    """Procesa en bloques y deduplica IDs globalmente con SQLite temporal."""
    meses = list(MESES if meses is None else meses)
    urls = URLS_VIAJES if urls is None else urls
    anterior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if anterior and anterior.get("data_version") != version_manifest(anterior):
        raise ValueError("El manifiesto está corrupto o fue modificado sin recalcular su versión.")
    congelado = anterior if not actualizar_version else None
    fuentes = {m: urls[m] for m in meses} | {"nomenclatura": nomenclatura_url}
    if congelado and (
        congelado["fuentes"] != fuentes
        or congelado["inicio"] != inicio
        or congelado["fin_exclusivo"] != fin
        or congelado["preprocesamiento"] != VERSION_PREPROCESAMIENTO
    ):
        raise ValueError("Cambió la configuración; usar --actualizar-version explícitamente.")
    for folder in (raw_dir, processed_dir, manifest_path.parent):
        folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "preprocesamiento": VERSION_PREPROCESAMIENTO,
        "inicio": inicio,
        "fin_exclusivo": fin,
        "fuentes": fuentes,
        "archivos": {},
        "salidas": {},
    }
    estaciones = set()
    totales = dict.fromkeys(
        ("originales", "invalidos", "fuera_periodo", "duplicados", "validos"), 0
    )
    with ExitStack() as stack:
        session = stack.enter_context(requests.Session())
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
                )
            ),
        )
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=processed_dir)))
        db = sqlite3.connect(tmp / "ids.db")
        stack.callback(db.close)
        db.execute("CREATE TABLE seen (id INTEGER PRIMARY KEY, contenido TEXT NOT NULL)")
        writer = pq.ParquetWriter(tmp / "viajes.parquet", SCHEMA, compression="zstd")
        stack.callback(writer.close)
        for mes in meses:
            path = raw_dir / f"datos_abiertos_{mes.replace('-', '_')}.csv"
            esperado = congelado["archivos"][mes]["sha256"] if congelado else None
            descargar(urls[mes], path, esperado, session, forzar=actualizar_version)
            encoding = detectar_codificacion(path)
            stats = dict.fromkeys(totales, 0)
            mes_inicio = pd.Timestamp(f"{mes}-01")
            mes_fin = mes_inicio + pd.offsets.MonthBegin(1)
            for bloque in pd.read_csv(path, encoding=encoding, chunksize=CHUNK_SIZE):
                limpio, contadores = normalizar_viajes(bloque, str(mes_inicio), str(mes_fin))
                for k, n in contadores.items():
                    stats[k] += n
                conservar = []
                for row in limpio.itertuples(index=False, name=None):
                    firma = json.dumps([str(x) for x in row[1:]], ensure_ascii=True)
                    previo = db.execute(
                        "SELECT contenido FROM seen WHERE id=?", (int(row[0]),)
                    ).fetchone()
                    if previo:
                        if previo[0] != firma:
                            raise ValueError(
                                f"Viaje {row[0]} repetido con contenido contradictorio."
                            )
                        conservar.append(False)
                    else:
                        db.execute("INSERT INTO seen VALUES (?,?)", (int(row[0]), firma))
                        conservar.append(True)
                stats["duplicados"] += len(limpio) - sum(conservar)
                limpio = limpio.loc[conservar]
                stats["validos"] += len(limpio)
                estaciones.update(map(int, limpio.origen_id))
                estaciones.update(map(int, limpio.destino_id))
                writer.write_table(
                    pa.Table.from_pandas(limpio, schema=SCHEMA, preserve_index=False)
                )
            db.commit()
            if not stats["validos"]:
                raise ValueError(f"El mes {mes} no contiene viajes válidos en su intervalo.")
            payload["archivos"][mes] = {
                "archivo": path.name,
                "sha256": sha256_archivo(path),
                "bytes": path.stat().st_size,
                "codificacion": encoding,
                **stats,
            }
            for k in totales:
                totales[k] += stats[k]
            logger.info(f"{mes}: {stats['validos']:,} viajes válidos")
        writer.close()
        if not totales["validos"]:
            raise ValueError("La ingesta no obtuvo viajes válidos.")
        path = raw_dir / "nomenclatura.csv"
        esperado = congelado["archivos"]["nomenclatura"]["sha256"] if congelado else None
        descargar(nomenclatura_url, path, esperado, session, forzar=actualizar_version)
        encoding = detectar_codificacion(path)
        nom = normalizar_estaciones(pd.read_csv(path, encoding=encoding))
        nom.to_parquet(tmp / "nomenclatura.parquet", index=False, compression="zstd")
        payload["archivos"]["nomenclatura"] = {
            "archivo": path.name,
            "sha256": sha256_archivo(path),
            "bytes": path.stat().st_size,
            "codificacion": encoding,
            "estaciones": len(nom),
        }
        for nombre in ("viajes.parquet", "nomenclatura.parquet"):
            payload["salidas"][nombre] = {
                "sha256": sha256_archivo(tmp / nombre),
                "filas": pq.read_metadata(tmp / nombre).num_rows,
            }
        payload["resumen"] = {
            **totales,
            "meses": len(meses),
            "estaciones_observadas": len(estaciones),
            "estaciones_sin_nomenclatura": sorted(estaciones - set(nom.id)),
            "estaciones_capacidad_invalida": int(nom.dpcapacity.isna().sum()),
        }
        payload["data_version"] = version_manifest(payload)
        if congelado and payload != congelado:
            raise ValueError(
                "El resultado no reproduce el manifiesto. Revisar entorno y transformación."
            )
        for nombre in payload["salidas"]:
            (tmp / nombre).replace(processed_dir / nombre)
        temporal = manifest_path.with_suffix(".json.part")
        temporal.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporal.replace(manifest_path)
    return payload


def registrar_ingesta_en_mlflow(resultado: dict, manifest_path=DATA_MANIFEST_PATH) -> None:
    """Registra la versión de datos en el run creado por ``mlflow run . -e ingest``."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # start_run() retoma el run del proyecto indicado en MLFLOW_RUN_ID.
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "stage_of_work": "ingest",
                "data_version": resultado["data_version"],
            }
        )
        mlflow.log_metrics(
            {
                k: v
                for k, v in resultado["resumen"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
        # Solo hashes, conteos y fuentes; los viajes no se suben a MLflow.
        mlflow.log_artifact(str(manifest_path), artifact_path="data")


@app.command()
def main(actualizar_version: bool = False):
    """Descarga, normaliza y verifica la versión de los datos configurados."""
    resultado = ingestar(actualizar_version=actualizar_version)
    logger.success(f"Datos: {resultado['data_version']} | {resultado['resumen']}")
    if os.environ.get("MLFLOW_RUN_ID"):
        registrar_ingesta_en_mlflow(resultado)


if __name__ == "__main__":
    app()
