"""Construye el panel horario sin cargar todos los viajes en memoria."""

import json

from loguru import logger
import pandas as pd
import pyarrow.parquet as pq

from mibici import features as f
from mibici.config import CHUNK_SIZE, DATA_MANIFEST_PATH, PROCESSED_DATA_DIR
from mibici.dataset import sha256_archivo, version_manifest


def verificar_datos(processed_dir=PROCESSED_DATA_DIR, manifest_path=DATA_MANIFEST_PATH):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["data_version"] != version_manifest(manifest):
        raise ValueError("Versión del manifiesto inválida.")
    for nombre, meta in manifest["salidas"].items():
        if sha256_archivo(processed_dir / nombre) != meta["sha256"]:
            raise ValueError(f"{nombre} no coincide con la versión del manifiesto.")
    return manifest


def preparar(processed_dir=PROCESSED_DATA_DIR, manifest_path=DATA_MANIFEST_PATH):
    manifest = verificar_datos(processed_dir, manifest_path)
    inicio, fin = pd.Timestamp(manifest["inicio"]), pd.Timestamp(manifest["fin_exclusivo"])
    acumulados = {"salidas": None, "llegadas": None}
    llegadas_fuera = 0
    for batch in pq.ParquetFile(processed_dir / "viajes.parquet").iter_batches(
        batch_size=CHUNK_SIZE
    ):
        viajes = batch.to_pandas()
        for nombre, estacion, tiempo in (
            ("salidas", "origen_id", "inicio_del_viaje"),
            ("llegadas", "destino_id", "fin_del_viaje"),
        ):
            periodo = viajes[tiempo].dt.floor("h")
            dentro = (periodo >= inicio) & (periodo < fin)
            if nombre == "llegadas":
                llegadas_fuera += int((~dentro).sum())
            serie = (
                viajes.loc[dentro].groupby([viajes.loc[dentro, estacion], periodo[dentro]]).size()
            )
            serie.index.names = ["estacion_id", "periodo"]
            anterior = acumulados[nombre]
            acumulados[nombre] = serie if anterior is None else anterior.add(serie, fill_value=0)
    conteos = pd.concat(acumulados, axis=1).fillna(0)
    ids = conteos.index.get_level_values("estacion_id").unique().sort_values()
    horas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    malla = pd.MultiIndex.from_product([ids, horas], names=["estacion_id", "periodo"])
    panel = conteos.reindex(malla, fill_value=0).astype("int32").reset_index()
    if panel.salidas.sum() != manifest["resumen"]["validos"]:
        raise ValueError("La agregación no conserva las salidas.")
    if panel.llegadas.sum() + llegadas_fuera != manifest["resumen"]["validos"]:
        raise ValueError("La agregación no conserva las llegadas dentro/fuera del periodo.")
    panel["flujo_neto"] = panel.llegadas - panel.salidas
    nom = pd.read_parquet(processed_dir / "nomenclatura.parquet")
    panel = f.agregar_variables_calendario(panel)
    panel = f.agregar_rezagos(panel)
    panel = f.agregar_promedios_moviles(panel)
    panel = f.agregar_capacidad(panel, nom)
    panel = f.agregar_flujo_acumulado_dia(panel)
    panel = f.construir_objetivo_riesgo(panel)
    # attrs son metadatos descriptivos; el entrenamiento no depende de su persistencia.
    panel.attrs = {
        "data_version": manifest["data_version"],
        "llegadas_fuera_periodo": llegadas_fuera,
    }
    panel.to_parquet(
        processed_dir / "panel_estacion_hora.parquet", index=False, compression="zstd"
    )
    logger.success(
        f"Panel: {len(panel):,} filas, {len(ids)} estaciones; llegadas fuera={llegadas_fuera}"
    )
    return panel


if __name__ == "__main__":
    preparar()
