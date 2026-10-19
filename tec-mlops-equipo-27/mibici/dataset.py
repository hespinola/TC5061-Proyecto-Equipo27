import typer
import csv
import hashlib
import json
import time
import pandas as pd
import unicodedata
import codecs
import requests
import re
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from datetime import datetime, timezone
from urllib.parse import urlparse

app = typer.Typer()

from mibici.config import (
    MESES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    URL_NOMENCLATURA,
    URLS_VIAJES,
    DATA_MANIFEST_PATH,
)

app = typer.Typer()

PIPELINE_VERSION = "1.0.0"
CHUNK_SIZE = 1 << 20  # 1 MiB
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mibici-dataset/1.0)"}

# Nombres normalizados
DATE_COLUMNS = ["inicio_del_viaje", "fin_del_viaje"]
INT_COLUMNS = ["viaje_id", "usuario_id", "origen_id", "destino_id", "ano_de_nacimiento"]
CATEGORY_COLUMNS = ["genero"]

# --------------------------------------------------------------------------- #
# Funciones de utilidad
# --------------------------------------------------------------------------- #
def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name
 
 
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()
 
 
def detect_encoding(path: Path) -> str:
    """utf-8 (con o sin BOM) si TODO el archivo decodifica bien; si no, latin-1.
 
    Cada mes puede venir en una codificación distinta, por eso se detecta por archivo.
    """
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"
 
 
def count_records(path: Path, encoding: str) -> int:
    """Número de filas de datos (sin contar el encabezado)."""
    with open(path, newline="", encoding=encoding) as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)
 
 
def download(url: str, dest: Path, force: bool = False, retries: int = 3) -> Path:
    """Descarga `url` a `dest` (escritura atómica). Omite si ya existe y no hay force."""
    if dest.exists() and not force:
        logger.info(f"{dest.name} ya existe, se omite la descarga.")
        return dest
 
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
 
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Descargando {url}")
            with requests.get(url, stream=True, timeout=60, headers=HEADERS) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc
            wait = 2**attempt
            logger.warning(f"Intento {attempt} falló ({exc}); reintentando en {wait}s")
            time.sleep(wait)
    return dest  # inalcanzable, por tipado
 
 
# --------------------------------------------------------------------------- #
# 1) Descarga
# --------------------------------------------------------------------------- #
def download_all(raw_dir: Path, force: bool = False) -> list[dict]:
    """Descarga viajes + nomenclatura. Devuelve una entrada de manifest por archivo raw."""
    files = []
 
    def entry(kind: str, periodo: str, url: str, path: Path, encoding: str) -> dict:
        return {
            "type": kind,
            "periodo": periodo,
            "filename": path.name,
            "url": url,
            "encoding": encoding,
            "size_bytes": path.stat().st_size,
            "records": count_records(path, encoding),
            "sha256": sha256_file(path),
        }
 
    for mes in tqdm(MESES, total=len(MESES), desc="Descargando viajes"):
        url = URLS_VIAJES[mes]
        path = download(url, raw_dir / filename_from_url(url), force=force)
        files.append(entry("viajes", mes, url, path, detect_encoding(path)))
 
    nom_path = download(URL_NOMENCLATURA, raw_dir / filename_from_url(URL_NOMENCLATURA), force=force)
    nom_periodo = nom_path.stem.removeprefix("nomenclatura_").replace("_", "-")
    files.append(entry("nomenclatura", nom_periodo, URL_NOMENCLATURA, nom_path, detect_encoding(nom_path)))
    return files
 
 
# --------------------------------------------------------------------------- #
# 2) Limpieza / normalización
# --------------------------------------------------------------------------- #
def normalize_column(name: str) -> str:
    """'Año_de_nacimiento' -> 'ano_de_nacimiento'; 'Inicio del viaje ' -> 'inicio_del_viaje'."""
    name = unicodedata.normalize("NFKD", str(name).strip())
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return name
 
 
def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes en columnas de texto (la codificación ya se resolvió al leer)."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df
 
 
def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [normalize_column(c) for c in df.columns]
    df = clean_strings(df)
 
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            n_bad = int(df[col].isna().sum())
            if n_bad:
                logger.warning(f"{col}: {n_bad:,} valores no se pudieron convertir a fecha (NaT)")
 
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
 
    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
 
    return df
 
 
def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [normalize_column(c) for c in df.columns]
    df = clean_strings(df)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    if not df["id"].is_unique:
        raise ValueError("La nomenclatura tiene ids de estación duplicados; el merge duplicaría viajes.")
    return df
 
 
def add_station_info(df: pd.DataFrame, stations: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Agrega nombre, lat, lon y capacidad de la estación `key` con el prefijo dado."""
    info = stations[["id", "name", "latitude", "longitude", "dpcapacity"]].rename(
        columns={
            "id": key,
            "name": f"{prefix}_nombre",
            "latitude": f"{prefix}_latitud",
            "longitude": f"{prefix}_longitud",
            "dpcapacity": f"{prefix}_capacidad",
        }
    )
    return df.merge(info, on=key, how="left")
 

def read_trips_file(path: Path) -> pd.DataFrame:
    """Lee un CSV de viajes con su codificación real y normaliza los nombres de columna."""
    df = pd.read_csv(path, encoding=detect_encoding(path))
    df.columns = [normalize_column(c) for c in df.columns]
    return df

 
def build_base_dataset(raw_dir: Path) -> pd.DataFrame:
    # Solo los archivos de MESES (evita leer archivos sueltos en raw/)
    viajes_paths = [raw_dir / filename_from_url(URLS_VIAJES[m]) for m in MESES]
    nom_path = raw_dir / filename_from_url(URL_NOMENCLATURA)
 
    stations = clean_stations(pd.read_csv(nom_path, encoding=detect_encoding(nom_path)))
 
    # Todos los CSV de viajes deben compartir estructura -> concat
    frames = [read_trips_file(p) for p in tqdm(viajes_paths, desc="Leyendo viajes")]
    ref_cols = list(frames[0].columns)
    for p, f in zip(viajes_paths, frames):
        if list(f.columns) != ref_cols:
            raise ValueError(f"{p.name} tiene columnas distintas: {list(f.columns)} vs {ref_cols}")
    data_df = clean_trips(pd.concat(frames, ignore_index=True))
    n_viajes = len(data_df)
 
    data_df = add_station_info(data_df, stations, key="origen_id", prefix="origen")
    data_df = add_station_info(data_df, stations, key="destino_id", prefix="destino")
 
    assert len(data_df) == n_viajes, "El merge cambió el número de filas"
    return data_df
 
 
def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
 
 
# --------------------------------------------------------------------------- #
# 3) Versionado: manifest + versión por contenido
# --------------------------------------------------------------------------- #
def compute_dataset_version(raw_files: list[dict]) -> str:
    """Hash estable de (pipeline, archivos raw y sus sha256).
 
    Cambia solo si cambian los datos de origen (un mes nuevo, un archivo republicado)
    o la lógica de limpieza (PIPELINE_VERSION). No depende de la fecha de ejecución.
    """
    h = hashlib.sha256()
    h.update(f"pipeline:{PIPELINE_VERSION}\n".encode())
    for f in sorted(raw_files, key=lambda f: f["filename"]):
        h.update(f"{f['filename']}:{f['sha256']}\n".encode())
    return h.hexdigest()[:12]
 
 
def write_manifest(raw_files: list[dict], df: pd.DataFrame, output_path: Path, manifest_path: Path) -> dict:
    manifest = {
        "version": compute_dataset_version(raw_files),
        "pipeline_version": PIPELINE_VERSION,
        "data_through": MESES[-1],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meses": list(MESES),
        "raw_files": raw_files,
        "processed": {
            "filename": output_path.name,
            "format": "parquet",
            "records": len(df),
            "size_bytes": output_path.stat().st_size,
            # Informativo: el binario puede variar entre versiones de pyarrow;
            # la identidad de los datos la da "version".
            "sha256": sha256_file(output_path),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
        },
    }
 
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prev.get("version") == manifest["version"]:
            logger.info(f"Datos sin cambios respecto al manifest anterior (version {manifest['version']}).")
        else:
            logger.warning(f"Los datos cambiaron: {prev.get('version')} -> {manifest['version']}")
 
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Manifest guardado en {manifest_path}")
    return manifest


# --------------------------------------------------------------------------- #
# 4) MLflow
# --------------------------------------------------------------------------- #
def read_data_version(manifest_path: Path = DATA_MANIFEST_PATH) -> str:
    """Para usar desde notebooks/entrenamiento: mlflow.log_param('data_version', read_data_version())."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))["version"]
 
 
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "mibici_viajes.parquet",
    manifest_path: Path = DATA_MANIFEST_PATH,
    force: bool = False,
    log_mlflow: bool = False,
):
    logger.info("Processing dataset...")
 
    raw_files = download_all(input_path, force=force)
 
    df = build_base_dataset(input_path)
    save_parquet(df, output_path)
    logger.info(f"{len(df):,} filas guardadas en {output_path}")
 
    manifest = write_manifest(raw_files, df, output_path, manifest_path)
 
    logger.success(f"Processing dataset complete. Versión de datos: {manifest['version']}")
 
 
if __name__ == "__main__":
    app()