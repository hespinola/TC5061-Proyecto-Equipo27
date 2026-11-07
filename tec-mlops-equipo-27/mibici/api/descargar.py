"""Descarga una versión del Model Registry para servirla sin servidor de MLflow.

    python -m mibici.api.descargar --destino modelo_p1

Usa las mismas variables que la API (``MLFLOW_TRACKING_URI``, ``MODELO_NOMBRE``,
``MODELO_VERSION`` o ``MODELO_ALIAS``). Guarda el modelo y ``modelo_info.json`` con
nombre, versión, alias y corrida, que la API reporta al servirlo con
``MODELO_RUTA=modelo_p1``.
"""

import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
import typer

from mibici.api.modelo import ARCHIVO_INFO, Ajustes, cargar, resolver_version

app = typer.Typer()


@app.command()
def main(destino: Path = Path("modelo_p1")):
    ajustes = Ajustes.desde_entorno()
    version, alias, run_id = resolver_version(ajustes)
    uri = f"models:/{ajustes.nombre}/{version}"
    if destino.exists():
        shutil.rmtree(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
    shutil.copytree(temporal, destino)
    info = {"nombre": ajustes.nombre, "version": version, "alias": alias, "run_id": run_id}
    (destino / ARCHIVO_INFO).write_text(json.dumps(info, indent=2) + "\n")
    # Comprueba que la copia se carga y cumple la firma antes de darla por buena.
    cargar(Ajustes(ruta_local=str(destino)))
    logger.success(f"{uri} guardado en {destino}")


if __name__ == "__main__":
    app()
