"""Monitoreo retrospectivo de las entradas de P1, sin modificar datos congelados."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici import drift
from mibici import features as f
from mibici.config import (
    DATA_MANIFEST_PATH,
    EQUIPO,
    FEATURES_P1,
    FIN_ENTRENAMIENTO,
    MES_NUEVO,
    MUESTRA_MONITOREO_P1,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    PROPORCION_MONITOREO_P1,
    REPORTS_DIR,
    REZAGOS_HORAS,
    SEED,
    UMBRAL_MONITOREO_P1,
    VENTANAS_MOVILES_HORAS,
)
from mibici.dataset import sha256_archivo
from mibici.preparar import preparar, verificar_datos

P1_CATEGORICAS = ("hora_del_dia", "dia_semana", "es_fin_de_semana", "mes")
P1_NUMERICAS = tuple(c for c in FEATURES_P1 if c not in P1_CATEGORICAS)
HORAS_CONTEXTO = max((*REZAGOS_HORAS, *VENTANAS_MOVILES_HORAS))
CLAVES = ["estacion_id", "periodo"]
CONTEOS = ["salidas", "llegadas"]


def crear_dataset_p1(frame, muestra=None):
    """Adapta las etiquetas para Evidently sobre una copia de las entradas de P1."""
    from evidently import DataDefinition, Dataset

    datos = drift.muestrear(frame[list(FEATURES_P1)], muestra).copy()
    for columna in P1_CATEGORICAS:
        datos[columna] = pd.Series(
            [None if pd.isna(valor) else str(valor) for valor in datos[columna]],
            index=datos.index,
            dtype=object,
        )
    definicion = DataDefinition(
        numerical_columns=list(P1_NUMERICAS),
        categorical_columns=list(P1_CATEGORICAS),
    )
    return Dataset.from_pandas(datos, data_definition=definicion)


def limites_mes(mes):
    inicio = pd.Timestamp(f"{mes}-01")
    if inicio.strftime("%Y-%m") != mes:
        raise ValueError("El mes debe tener formato AAAA-MM.")
    return inicio, inicio + pd.offsets.MonthBegin(1)


def _conteos(viajes, columna_estacion, columna_fecha, nombre, inicio, fin):
    periodos = viajes[columna_fecha].dt.floor("h")
    dentro = (periodos >= inicio) & (periodos < fin)
    conteos = viajes.loc[dentro].groupby(
        [viajes.loc[dentro, columna_estacion], periodos.loc[dentro]]
    ).size()
    conteos.index.names = CLAVES
    return conteos.rename(nombre)


def construir_actual(nuevo, contexto, arrastres, inicio, fin):
    """Construye octubre para las estaciones originales y conserva las horas en cero.

    contexto contiene las últimas horas del panel congelado. arrastres contiene
    llegadas del histórico que ocurren dentro del mes actual. Las estaciones nuevas
    se cuentan por separado y no reciben una historia artificial llena de ceros.
    """
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    comienzo_contexto = inicio - pd.Timedelta(hours=HORAS_CONTEXTO)
    contexto = contexto.loc[
        (contexto.periodo >= comienzo_contexto) & (contexto.periodo < inicio),
        CLAVES + CONTEOS,
    ].copy()
    if contexto.empty or contexto.duplicated(CLAVES).any():
        raise ValueError("El contexto histórico está vacío o tiene claves duplicadas.")
    ids = sorted(contexto.estacion_id.unique().tolist())
    horas_contexto = pd.date_range(comienzo_contexto, inicio, freq="h", inclusive="left")
    malla_contexto = pd.MultiIndex.from_product([ids, horas_contexto], names=CLAVES)
    contexto = contexto.set_index(CLAVES).reindex(malla_contexto)
    if contexto[CONTEOS].isna().any().any():
        raise ValueError("Faltan horas de contexto; no se pueden calcular los rezagos.")
    contexto = contexto.reset_index()

    salidas = _conteos(nuevo, "origen_id", "inicio_del_viaje", "salidas", inicio, fin)
    llegadas = _conteos(nuevo, "destino_id", "fin_del_viaje", "llegadas", inicio, fin)
    llegadas_previas = _conteos(
        arrastres, "destino_id", "fin_del_viaje", "llegadas", inicio, fin
    )
    llegadas = llegadas.add(llegadas_previas, fill_value=0)
    horas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    malla = pd.MultiIndex.from_product([ids, horas], names=CLAVES)
    actual = (
        pd.concat([salidas, llegadas], axis=1)
        .reindex(malla)
        .fillna(0)
        .astype("int32")
        .reset_index()
    )
    salidas_esperadas = int(
        (
            nuevo.origen_id.isin(ids)
            & nuevo.inicio_del_viaje.between(inicio, fin, inclusive="left")
        ).sum()
    )
    llegadas_esperadas = sum(
        int(
            (
                viajes.destino_id.isin(ids)
                & viajes.fin_del_viaje.between(inicio, fin, inclusive="left")
            ).sum()
        )
        for viajes in (nuevo, arrastres)
    )
    if int(actual.salidas.sum()) != salidas_esperadas:
        raise ValueError("El panel actual no conserva las salidas de la cohorte.")
    if int(actual.llegadas.sum()) != llegadas_esperadas:
        raise ValueError("El panel actual no conserva las llegadas de la cohorte.")

    panel = pd.concat([contexto, actual], ignore_index=True)
    panel["flujo_neto"] = panel.llegadas - panel.salidas
    panel = f.agregar_variables_calendario(panel)
    panel = f.agregar_rezagos(panel)
    panel = f.agregar_promedios_moviles(panel)
    panel = panel.loc[
        (panel.periodo >= inicio) & (panel.periodo < fin)
    ].reset_index(drop=True)

    ids_actuales = set(nuevo.origen_id) | set(nuevo.destino_id)
    cobertura = {
        "estaciones_cohorte": len(ids),
        "estaciones_nuevas_fuera_cohorte": sorted(map(int, ids_actuales - set(ids))),
        "estaciones_cohorte_sin_viajes_del_mes": sorted(map(int, set(ids) - ids_actuales)),
        "salidas_en_cohorte": salidas_esperadas,
        "salidas_fuera_cohorte": int(len(nuevo) - salidas_esperadas),
        "llegadas_historicas_en_mes": int(len(arrastres)),
        "filas_estacion_hora": len(panel),
        "horas_por_estacion": len(horas),
    }
    return panel, cobertura


def _buscar_run(objeto, run_id):
    if isinstance(objeto, dict):
        if objeto.get("run_id") == run_id and "metricas" in objeto:
            return objeto
        valores = objeto.values()
    elif isinstance(objeto, list):
        valores = objeto
    else:
        return None
    for valor in valores:
        encontrado = _buscar_run(valor, run_id)
        if encontrado is not None:
            return encontrado
    return None


def _modelo_documentado(manifest):
    registro = json.loads((REPORTS_DIR / "registry_results.json").read_text(encoding="utf-8"))
    nombre = f"mibici-{EQUIPO}-p1-demanda"
    modelos = [m for m in registro["models"] if m["name"] == nombre]
    if len(modelos) != 1:
        raise ValueError("No se identificó una única versión documentada de P1.")
    modelo = modelos[0]
    if modelo["data_version"] != manifest["data_version"]:
        raise ValueError("El modelo documentado y el manifiesto tienen distintas versiones de datos.")
    experimentos = json.loads(
        (REPORTS_DIR / "experiment_results.json").read_text(encoding="utf-8")
    )
    resultado = _buscar_run(experimentos, modelo["run_id"])
    if resultado is None or resultado.get("modelo") != "hgb":
        raise ValueError("Este monitoreo requiere identificar el run HGB de P1 y sus métricas.")
    return modelo, resultado


def _guardar_grafica(tabla, destino, umbral):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    grupos = (
        ("numérica", "Wasserstein normalizada"),
        ("categórica", "Jensen-Shannon"),
    )
    for eje, (tipo, titulo) in zip(ejes, grupos, strict=True):
        datos = tabla.loc[tabla.tipo == tipo].sort_values("puntaje")
        eje.barh(
            datos.index,
            datos.puntaje,
            color=["tab:red" if v else "tab:green" for v in datos.drift],
        )
        eje.axvline(umbral, color="black", linestyle="--", label="umbral")
        eje.set_title(titulo)
        eje.set_xlabel("Distancia")
        eje.legend()
    fig.suptitle("P1: drift de entradas respecto al entrenamiento")
    fig.tight_layout()
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)


def ejecutar(
    mes=MES_NUEVO,
    muestra=MUESTRA_MONITOREO_P1,
    umbral=UMBRAL_MONITOREO_P1,
    proporcion=PROPORCION_MONITOREO_P1,
    desde_csv=False,
    destino=None,
):
    """Genera reportes y evidencia; no entrena modelos ni consulta un alias mutable.

    Con ``desde_csv`` el histórico se reconstruye desde los CSV congelados del
    manifiesto (misma ruta que el A/B), sin requerir los Parquet de la ingesta.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    if muestra is not None and (isinstance(muestra, bool) or muestra < 1):
        raise ValueError("La muestra debe ser un entero positivo o None.")
    if not np.isfinite(umbral) or umbral <= 0 or not 0 < proporcion <= 1:
        raise ValueError("Umbral y proporción de drift fuera de rango.")
    inicio, fin = limites_mes(mes)
    if desde_csv:
        manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = verificar_datos()
    hash_manifiesto = sha256_archivo(DATA_MANIFEST_PATH)
    if inicio != pd.Timestamp(manifest["fin_exclusivo"]):
        raise ValueError(
            "El mes actual debe ser contiguo al histórico congelado para disponer de contexto."
        )
    if inicio <= pd.Timestamp(FIN_ENTRENAMIENTO):
        raise ValueError("El mes actual debe ser posterior al entrenamiento.")
    modelo, resultado_modelo = _modelo_documentado(manifest)

    drift.descargar_mes(mes)
    nuevo, stats = drift.cargar_mes(mes)
    dias = set(nuevo.inicio_del_viaje.dt.strftime("%Y-%m-%d"))
    esperados = set(pd.date_range(inicio, fin, freq="D", inclusive="left").strftime("%Y-%m-%d"))
    if dias != esperados:
        raise ValueError(
            f"Revisar cobertura del CSV: días sin viajes registrados={sorted(esperados - dias)}."
        )

    if desde_csv:
        # Import diferido: ab_testing importa este módulo.
        from mibici.ab_testing import historico_desde_csv

        historico, arrastres_csv = historico_desde_csv(manifest)
    else:
        # La preparación original se ejecuta con la escritura del panel desactivada.
        historico = preparar(guardar=False)
    corte = pd.Timestamp(FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)
    referencia_base = historico.loc[
        historico.periodo < corte, CLAVES + ["salidas", *FEATURES_P1]
    ].copy()
    referencia = referencia_base.dropna(subset=[*FEATURES_P1, "salidas"]).reset_index(drop=True)
    excluidas_ref = len(referencia_base) - len(referencia)
    del referencia_base
    esperado = int(resultado_modelo["metricas"]["train_filas"])
    if len(referencia) != esperado:
        raise ValueError(
            f"La referencia tiene {len(referencia)} filas; el run documenta {esperado}."
        )

    contexto = historico.loc[
        historico.periodo >= inicio - pd.Timedelta(hours=HORAS_CONTEXTO),
        CLAVES + CONTEOS,
    ].copy()
    del historico
    if desde_csv:
        arrastres = arrastres_csv.loc[
            (arrastres_csv.fin_del_viaje >= inicio) & (arrastres_csv.fin_del_viaje < fin)
        ].reset_index(drop=True)
    else:
        arrastres = pd.read_parquet(
            PROCESSED_DATA_DIR / "viajes.parquet",
            columns=["fin_del_viaje", "destino_id"],
            filters=[("fin_del_viaje", ">=", inicio), ("fin_del_viaje", "<", fin)],
        )
    actual_base, cobertura = construir_actual(nuevo, contexto, arrastres, inicio, fin)
    faltantes_actual = actual_base[list(FEATURES_P1)].isna().sum()
    actual = actual_base.dropna(subset=[*FEATURES_P1, "salidas"]).reset_index(drop=True)
    if actual.empty:
        raise ValueError("No hay filas elegibles del mes actual.")
    if not np.isfinite(referencia[list(FEATURES_P1)].to_numpy(dtype=float)).all():
        raise ValueError("La referencia contiene predictores no finitos.")
    if not np.isfinite(actual[list(FEATURES_P1)].to_numpy(dtype=float)).all():
        raise ValueError("El mes actual contiene predictores no finitos.")
    mes_control = f"{inicio.year - 1:04d}-{inicio.month:02d}"
    control = referencia.loc[referencia.periodo.dt.strftime("%Y-%m") == mes_control]
    if control.empty:
        raise ValueError("No existe el mismo mes del año anterior dentro de la referencia.")

    ds_ref = crear_dataset_p1(referencia, muestra)
    ds_actual = crear_dataset_p1(actual, muestra)
    ds_control = crear_dataset_p1(control, muestra)
    destino = REPORTS_DIR / "drift" if destino is None else Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    archivos = {}
    calidad_path = destino / f"calidad_p1_{mes}_vs_entrenamiento.html"
    Report([DataSummaryPreset()]).run(ds_actual, ds_ref).save_html(str(calidad_path))
    archivos["calidad_html"] = calidad_path

    def comparar(ds_comparacion, sufijo):
        snapshot = Report(
            [
                DataDriftPreset(
                    num_method="wasserstein",
                    cat_method="jensenshannon",
                    threshold=umbral,
                    drift_share=proporcion,
                )
            ],
            include_tests=True,
        ).run(ds_actual, ds_comparacion)
        html = destino / f"drift_p1_{mes}_{sufijo}.html"
        csv = destino / f"drift_p1_{mes}_{sufijo}.csv"
        snapshot.save_html(str(html))
        tabla = drift.tabla_drift(snapshot.dict(), numericas=P1_NUMERICAS)
        tabla.to_csv(csv)
        archivos[f"{sufijo}_html"] = html
        archivos[f"{sufijo}_csv"] = csv
        return tabla

    tabla = comparar(ds_ref, "vs_entrenamiento")
    tabla_control = comparar(ds_control, f"vs_{mes_control}")
    figura = destino / f"drift_p1_{mes}.png"
    _guardar_grafica(tabla, figura, umbral)
    archivos["grafica_png"] = figura
    if not desde_csv:
        verificar_datos()
    if sha256_archivo(DATA_MANIFEST_PATH) != hash_manifiesto:
        raise ValueError("El manifiesto cambió durante la ejecución.")

    fuentes_codigo = [
        PROJ_ROOT / "mibici" / f"{nombre}.py"
        for nombre in ("config", "dataset", "features", "preparar", "drift", "monitoreo_p1")
    ]
    resumen = {
        "ejercicio": "monitoreo_retrospectivo_p1",
        "modelo_documentado": {
            k: modelo[k] for k in ("name", "version", "run_id", "source_commit", "data_version")
        },
        "hiperparametros_modelo": resultado_modelo["hp"],
        "data_version_referencia": manifest["data_version"],
        "sha256_manifiesto": hash_manifiesto,
        "preparacion": "csv_congelados_solo_p1" if desde_csv else "parquet_ingesta_completa",
        "archivo_actual": stats,
        "periodos": {
            "referencia_inicio": manifest["inicio"],
            "referencia_fin_exclusivo": str(corte.date()),
            "referencia_primera_fila_elegible": referencia.periodo.min().isoformat(),
            "actual_inicio": str(inicio.date()),
            "actual_fin_exclusivo": str(fin.date()),
            "control_estacional": mes_control,
            "horas_contexto": HORAS_CONTEXTO,
        },
        "filas": {
            "referencia_elegibles": len(referencia),
            "referencia_run_documentado": esperado,
            "referencia_excluidas_por_rezagos": excluidas_ref,
            "actual_antes_filtrar": len(actual_base),
            "actual_elegibles": len(actual),
            "actual_excluidas": len(actual_base) - len(actual),
            "control_elegibles": len(control),
            "muestra_referencia": min(len(referencia), muestra or len(referencia)),
            "muestra_actual": min(len(actual), muestra or len(actual)),
            "muestra_control": min(len(control), muestra or len(control)),
        },
        "calidad": {
            "dias_con_viajes": len(dias),
            "dias_esperados": len(esperados),
            "faltantes_actual_por_variable": {
                col: int(valor) for col, valor in faltantes_actual.items()
            },
            **cobertura,
        },
        "configuracion": {
            "features": list(FEATURES_P1),
            "categoricas": list(P1_CATEGORICAS),
            "numericas": list(P1_NUMERICAS),
            "metodo_numerico": "wasserstein",
            "metodo_categorico": "jensenshannon",
            "umbral": umbral,
            "proporcion_alerta": proporcion,
            "seed": SEED,
            "muestra_solicitada": muestra,
        },
        "variables_con_drift": int(tabla.drift.sum()),
        "proporcion_variables_con_drift": float(tabla.drift.mean()),
        "drift_por_variable": tabla.reset_index().to_dict(orient="records"),
        "control_por_variable": tabla_control.reset_index().to_dict(orient="records"),
        "entorno": {
            paquete: version(paquete)
            for paquete in ("evidently", "pandas", "numpy", "pyarrow", "scikit-learn")
        },
        "codigo_sha256": {
            str(path.relative_to(PROJ_ROOT)): sha256_archivo(path) for path in fuentes_codigo
        },
        "archivos": {nombre: _ruta_relativa(path) for nombre, path in archivos.items()},
        "limitaciones": [
            "Se monitorea la cohorte de estaciones del panel original; las nuevas se reportan aparte.",
            "Una hora sin viajes observados se representa con cero como en el pipeline original.",
            "El control de un mismo mes en otro año no identifica por sí solo una causa estacional.",
            "La referencia estacional pierde el calentamiento inicial de los rezagos.",
            "Los umbrales son exploratorios; el drift no demuestra pérdida de desempeño.",
            "La versión base se identifica por evidencia versionada, sin resolver un alias vigente.",
        ],
    }
    resumen_path = destino / f"resumen_p1_{mes}.json"
    resumen_path.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Modelo documentado: {modelo['name']} v{modelo['version']}")
    print(f"Referencia: {len(referencia):,} filas; coincide con el run documentado.")
    print(f"Actual: {len(actual):,} filas; {len(dias)} días con viajes registrados.")
    print(f"Variables con drift: {int(tabla.drift.sum())} de {len(tabla)}")
    print(tabla.to_string())
    print(f"Evidencia: {resumen_path}")
    return resumen


def _ruta_relativa(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJ_ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mes", default=MES_NUEVO)
    parser.add_argument("--muestra", type=int, default=MUESTRA_MONITOREO_P1)
    parser.add_argument(
        "--desde-csv",
        action="store_true",
        help="Reconstruye el histórico desde los CSV del manifiesto, sin Parquet congelados.",
    )
    parser.add_argument(
        "--destino",
        default=None,
        help="Carpeta de salida (por defecto reports/drift).",
    )
    args = parser.parse_args()
    ejecutar(
        mes=args.mes,
        muestra=None if args.muestra == 0 else args.muestra,
        desde_csv=args.desde_csv,
        destino=args.destino,
    )


if __name__ == "__main__":
    main()
