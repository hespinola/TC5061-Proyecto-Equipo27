"""Configuración central del proyecto MiBici (equipo 27).

Todas las rutas, fechas, semillas, fuentes de datos y parámetros de MLflow viven
aquí. Ningún otro archivo del módulo debe repetir estos valores: deben importarlos
desde ``mibici.config``.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"
# Manifiesto versionado en Git con hash SHA-256 y número de registros por archivo.
DATA_MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"

# ---------------------------------------------------------------------------
# Equipo y reproducibilidad
# ---------------------------------------------------------------------------
EQUIPO = "equipo27"
SEED = 42

# Feature engineering: predicción al inicio de t, con observaciones anteriores a t.
# La ventana futura de P2 no es la anticipación de las variables predictoras.
REZAGOS_HORAS = (24, 168)
VENTANAS_MOVILES_HORAS = (24, 168)
VENTANA_RIESGO_HORAS = 3
UMBRAL_RIESGO = 0.5
VERSION_PREPROCESAMIENTO = "1"
K_GRUPOS_CANDIDATOS = (2, 3, 4, 5, 6)
MUESTRA_SILHOUETTE = 5000
UMBRALES_CLASIFICACION = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5)
CHUNK_SIZE = 100_000
TIMEOUT_DESCARGA = (15, 120)
PORTAL_DATOS_URL = "https://mibici.net/es/datos-abiertos/"
# Cabeceras explícitas para servidores que rechazan el cliente HTTP por defecto.
# No son credenciales ni modifican la validación TLS o los hashes de los archivos.
CABECERAS_DESCARGA = {
    "User-Agent": "Mozilla/5.0 (compatible; MiBiciEquipo27/1.0)",
    "Accept": "text/csv,application/csv,application/octet-stream,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Referer": PORTAL_DATOS_URL,
}

# ---------------------------------------------------------------------------
# Periodo de datos (propuesta del equipo, formato AAAA-MM, ambos inclusive)
# ---------------------------------------------------------------------------
# Justificación: los 12 meses más recientes cubren un ciclo anual completo
# (temporada de lluvias, vacaciones y regreso a clases) y son los más
# consistentes con la nomenclatura de estaciones de 2026-09.
PERIODO_INICIO = "2025-10"
PERIODO_FIN = "2026-09"

# Partición temporal (sin barajar) para no usar información del futuro:
#   entrenamiento: inicio de PERIODO_INICIO .. FIN_ENTRENAMIENTO
#   validación:    día siguiente            .. FIN_VALIDACION
#   prueba:        día siguiente            .. fin de PERIODO_FIN
FIN_ENTRENAMIENTO = "2026-06-30"
FIN_VALIDACION = "2026-07-31"

# ---------------------------------------------------------------------------
# Fuentes de datos: https://mibici.net/es/datos-abiertos/
# ---------------------------------------------------------------------------
# Las URLs de MiBici incluyen un identificador de carpeta que no se puede
# deducir del mes, por eso se listan de forma explícita.
URLS_VIAJES = {
    "2025-10": "https://mibici.net/site/assets/files/115104/datos_abiertos_2025_10.csv",
    "2025-11": "https://mibici.net/site/assets/files/116773/datos_abiertos_2025_11.csv",
    "2025-12": "https://mibici.net/site/assets/files/118153/datos_abiertos_2025_12.csv",
    "2026-01": "https://mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "2026-02": "https://mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "2026-03": "https://mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "2026-04": "https://mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "2026-05": "https://mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "2026-06": "https://mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "2026-07": "https://mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "2026-08": "https://mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "2026-09": "https://mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv",
}
URL_NOMENCLATURA = "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"

# Meses del periodo en orden; dataset.py debe descargar exactamente estos.
MESES = sorted(m for m in URLS_VIAJES if PERIODO_INICIO <= m <= PERIODO_FIN)
OBSERVACION_INICIO = f"{PERIODO_INICIO}-01"
_anio, _mes = map(int, PERIODO_FIN.split("-"))
OBSERVACION_FIN = f"{_anio + (_mes == 12):04d}-{_mes % 12 + 1:02d}-01"

# ---------------------------------------------------------------------------
# MLflow Tracking
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "mibici-equipo27"
# Por defecto, una base SQLite local (excluida de Git). Se puede cambiar con la
# variable de entorno MLFLOW_TRACKING_URI en un archivo .env.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
