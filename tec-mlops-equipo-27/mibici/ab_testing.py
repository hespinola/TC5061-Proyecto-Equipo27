"""Comparación retrospectiva de dos versiones registradas de P1, sin promover aliases.

A reconstruye el HGB documentado; B conserva sus hiperparámetros y amplía el
entrenamiento hasta el 14 de octubre. La evaluación común es del 15 al 31.
"""

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import mlflow
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mibici import drift
from mibici import features as f
from mibici.config import (
    DATA_MANIFEST_PATH,
    FEATURES_P1,
    FIN_ENTRENAMIENTO,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEED,
)
from mibici.dataset import descargar, sha256_archivo, version_manifest
from mibici.monitoreo_p1 import HORAS_CONTEXTO, _modelo_documentado, construir_actual
from mibici.preparar import preparar, verificar_datos

INICIO_ACTUAL = pd.Timestamp("2026-10-01")
INICIO_EVALUACION = pd.Timestamp("2026-10-15")
FIN_ACTUAL = pd.Timestamp("2026-11-01")
MEJORA_MINIMA = 0.01
COLUMNAS = ["estacion_id", "periodo", *FEATURES_P1, "salidas"]


def historico_desde_csv(manifest, raw_dir=RAW_DATA_DIR):
    """Reconstruye solo P1 desde los CSV congelados, sin necesitar atributos de estaciones.

    Verifica hashes, tamaños, limpieza por mes y unicidad global. No genera ni
    modifica los Parquet/manifiesto de la ingesta completa (que también sirve P2/P3).
    """
    if version_manifest(manifest) != manifest["data_version"]:
        raise ValueError("Manifiesto inválido.")
    frames = []
    with requests.Session() as session:
        for mes, url in sorted(manifest["fuentes"].items()):
            if mes == "nomenclatura":
                continue
            meta = manifest["archivos"][mes]
            path = raw_dir / meta["archivo"]
            descargar(url, path, meta["sha256"], session)
            frame, stats = drift.cargar_mes(mes, raw_dir=raw_dir)
            if stats != meta:
                raise ValueError(f"{mes}: la limpieza no reproduce el manifiesto.")
            frames.append(frame)
    viajes = pd.concat(frames, ignore_index=True)
    del frames
    if viajes.viaje_id.duplicated().any() or len(viajes) != manifest["resumen"]["validos"]:
        raise ValueError("Los viajes no reproducen el conteo o unicidad global congelados.")
    inicio, fin = pd.Timestamp(manifest["inicio"]), pd.Timestamp(manifest["fin_exclusivo"])
    panel = f.construir_panel_estacion_periodo(viajes)
    panel = panel.loc[(panel.periodo >= inicio) & (panel.periodo < fin)].copy()
    if panel.estacion_id.nunique() != manifest["resumen"]["estaciones_observadas"]:
        raise ValueError("La cohorte de estaciones difiere de la documentada.")
    horas = len(pd.date_range(inicio, fin, freq="h", inclusive="left"))
    if len(panel) != panel.estacion_id.nunique() * horas:
        raise ValueError("El panel no cubre todas las horas congeladas.")
    if panel.salidas.sum() != len(viajes):
        raise ValueError("El panel no conserva salidas históricas.")
    arrastres = viajes.loc[
        (viajes.fin_del_viaje >= fin) & (viajes.fin_del_viaje < FIN_ACTUAL),
        ["fin_del_viaje", "destino_id"],
    ].copy()
    del viajes
    panel = f.agregar_variables_calendario(panel)
    panel = f.agregar_rezagos(panel)
    panel = f.agregar_promedios_moviles(panel)
    return panel, arrastres


def huella(frame):
    """Identifica todas las filas, predictores y objetivos, en su orden efectivo."""
    valores = pd.util.hash_pandas_object(frame[COLUMNAS], index=False).to_numpy()
    return hashlib.sha256(valores.tobytes()).hexdigest()


def particionar(historico, actual):
    """Cortes fijados antes de evaluar; no se ajustan con resultados del holdout."""
    for nombre, frame in (("historico", historico), ("actual", actual)):
        if frame.empty or frame.duplicated(["estacion_id", "periodo"]).any():
            raise ValueError(f"{nombre}: filas vacías o claves duplicadas.")
        if not pd.api.types.is_datetime64_any_dtype(frame.periodo):
            raise ValueError(f"{nombre}: periodo debe ser datetime.")
    if historico.periodo.max() >= INICIO_ACTUAL:
        raise ValueError("El histórico invade el periodo actual.")
    if actual.periodo.min() < INICIO_ACTUAL or actual.periodo.max() >= FIN_ACTUAL:
        raise ValueError("El periodo actual debe limitarse a octubre de 2026.")
    horas = pd.date_range(INICIO_ACTUAL, FIN_ACTUAL, freq="h", inclusive="left")
    esperado = pd.MultiIndex.from_product(
        [sorted(historico.estacion_id.unique()), horas], names=["estacion_id", "periodo"]
    )
    if not actual.set_index(["estacion_id", "periodo"]).index.sort_values().equals(esperado):
        raise ValueError(
            "La evaluación requiere la cohorte completa y todas las horas de octubre."
        )
    historico = historico[COLUMNAS].dropna().sort_values(["estacion_id", "periodo"])
    actual = actual[COLUMNAS].sort_values(["estacion_id", "periodo"])
    for frame in (historico, actual):
        if not np.isfinite(frame[[*FEATURES_P1, "salidas"]].to_numpy(dtype=float)).all():
            raise ValueError("Hay predictores u objetivos no finitos.")
        if (frame.salidas < 0).any():
            raise ValueError("La demanda observada no puede ser negativa.")
    a = historico.loc[historico.periodo < pd.Timestamp(FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)]
    b = pd.concat([historico, actual.loc[actual.periodo < INICIO_EVALUACION]], ignore_index=True)
    evaluacion = actual.loc[actual.periodo >= INICIO_EVALUACION].reset_index(drop=True)
    if a.empty or b.empty or evaluacion.empty:
        raise ValueError("Partición vacía.")
    return a.reset_index(drop=True), b, evaluacion


def decidir(a, b):
    """B necesita >=1% de mejora MAE sin empeorar RMSE; en otro caso se conserva A."""
    for metricas in (a, b):
        if any(not np.isfinite(metricas[k]) or metricas[k] < 0 for k in ("mae", "rmse")):
            raise ValueError("Métricas inválidas para decidir.")
    mejora = (a["mae"] - b["mae"]) / a["mae"] if a["mae"] > 0 else 0.0
    ganador = "B" if mejora >= MEJORA_MINIMA and b["rmse"] <= a["rmse"] else "A"
    return {
        "recomendacion": ganador,
        "mejora_relativa_mae_b": mejora,
        "regla": "B si mejora MAE al menos 1% y no empeora RMSE; en otro caso conservar A",
        "promocion_automatica": False,
    }


def predecir(modelo, evaluacion):
    salida = np.asarray(modelo.predict(evaluacion[list(FEATURES_P1)].astype(float)), dtype=float)
    if salida.ndim != 1 or len(salida) != len(evaluacion) or not np.isfinite(salida).all():
        raise ValueError("Predicciones inválidas.")
    # Mismo recorte y redondeo que el servicio de la API para ambos brazos.
    return np.array([round(float(x), 4) for x in np.clip(salida, 0, None)])


def comparar_registrados(nombre, versiones, evaluacion):
    resultados, predicciones = {}, {}
    for brazo, numero in versiones.items():
        uri = f"models:/{nombre}/{numero}"
        modelo = mlflow.pyfunc.load_model(uri)
        if modelo.metadata.get_input_schema().input_names() != list(FEATURES_P1):
            raise ValueError("Firma incompatible con P1.")
        pred = predecir(modelo, evaluacion)
        predicciones[brazo] = pred
        resultados[brazo] = {
            "version": str(numero),
            "uri": uri,
            "mae": float(mean_absolute_error(evaluacion.salidas, pred)),
            "rmse": float(root_mean_squared_error(evaluacion.salidas, pred)),
            "filas": len(evaluacion),
            "evaluacion_sha256": huella(evaluacion),
        }
    diario = []
    for dia, indices in evaluacion.groupby(
        evaluacion.periodo.dt.strftime("%Y-%m-%d")
    ).indices.items():
        fila = {"dia": dia, "filas": len(indices)}
        for brazo in versiones:
            fila[f"mae_{brazo}"] = float(
                mean_absolute_error(evaluacion.salidas.iloc[indices], predicciones[brazo][indices])
            )
            fila[f"rmse_{brazo}"] = float(
                root_mean_squared_error(
                    evaluacion.salidas.iloc[indices], predicciones[brazo][indices]
                )
            )
        diario.append(fila)
    return resultados, diario


def entrenar_registrar(a, b, evaluacion, hp, nombre, trazabilidad, destino):
    """Usa el registry configurado por el llamador y registra dos versiones distintas."""
    if any(frame.periodo.max() >= evaluacion.periodo.min() for frame in (a, b)):
        raise ValueError("Entrenamiento y evaluación se solapan temporalmente.")
    if destino.exists():
        raise FileExistsError("La evidencia ya existe; no sobrescribir una evaluación anterior.")
    cliente = mlflow.MlflowClient()
    versiones, corridas = {}, {}
    protocolo = {
        "evaluacion_inicio": str(evaluacion.periodo.min()),
        "evaluacion_fin": str(evaluacion.periodo.max()),
        "evaluacion_sha256": huella(evaluacion),
        "regla": decidir({"mae": 1.0, "rmse": 1.0}, {"mae": 1.0, "rmse": 1.0})["regla"],
        "hiperparametros": hp,
        "semilla": SEED,
        "inferencia": "inicio de cada hora con observaciones anteriores; recorte a 0 y 4 decimales",
    }
    with mlflow.start_run(run_name="P1-comparacion-AB-retrospectiva") as padre:
        mlflow.log_dict(protocolo, "protocolo.json")
        mlflow.log_dict(trazabilidad, "trazabilidad.json")
        for brazo, frame in (("A", a), ("B", b)):
            with mlflow.start_run(run_name=f"P1-{brazo}", nested=True) as corrida:
                corridas[brazo] = corrida.info.run_id
                modelo = HistGradientBoostingRegressor(
                    max_iter=int(hp["n_estimators"]),
                    max_depth=int(hp["max_depth"]),
                    learning_rate=float(hp["learning_rate"]),
                    random_state=SEED,
                    early_stopping=False,
                    categorical_features=None,
                )
                X = frame[list(FEATURES_P1)].astype(float)
                entrenamiento_hash = huella(frame)
                mlflow.log_params(
                    {
                        **hp,
                        "seed": SEED,
                        "train_filas": len(frame),
                        "train_inicio": str(frame.periodo.min()),
                        "train_fin": str(frame.periodo.max()),
                    }
                )
                mlflow.set_tags(
                    {
                        "brazo": brazo,
                        "train_sha256": entrenamiento_hash,
                        "baseline_reconstruido": "true",
                        "evaluacion_sha256": huella(evaluacion),
                    }
                )
                # Fuente completa identificada por hash; no se suben viajes ni datos personales.
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        frame[COLUMNAS],
                        source=str(DATA_MANIFEST_PATH),
                        digest=entrenamiento_hash[:32],
                        targets="salidas",
                        name=f"P1-entrenamiento-{brazo}",
                    ),
                    context="training",
                )
                print(
                    f"Entrenando {brazo}: {len(frame):,} filas hasta {frame.periodo.max()}",
                    flush=True,
                )
                modelo.fit(X, frame.salidas.astype(float))
                ejemplo = X.head(5)
                info = mlflow.sklearn.log_model(
                    modelo,
                    name="model",
                    input_example=ejemplo,
                    serialization_format="cloudpickle",
                    signature=mlflow.models.infer_signature(ejemplo, modelo.predict(ejemplo)),
                    pip_requirements=[
                        f"{p}=={version(p)}"
                        for p in ("scikit-learn", "numpy", "pandas", "cloudpickle", "mlflow")
                    ],
                )
                mv = mlflow.register_model(info.model_uri, nombre)
                versiones[brazo] = str(mv.version)
                print(f"Registrado {brazo}: {nombre} versión {mv.version}", flush=True)
                cliente.update_model_version(
                    nombre,
                    mv.version,
                    description=(
                        f"A/B retrospectivo: brazo {brazo}. "
                        + (
                            "Reconstrucción local del HGB documentado; no es el artefacto original. "
                            if brazo == "A"
                            else "Reentrenado con historia ampliada. "
                        )
                        + f"Entrenamiento hasta {frame.periodo.max()}; hash {entrenamiento_hash}."
                    ),
                )
                for clave, valor in {
                    "brazo": brazo,
                    "train_sha256": entrenamiento_hash,
                    "baseline_reconstruido": "true",
                }.items():
                    cliente.set_model_version_tag(nombre, mv.version, clave, valor)
                recargado = mlflow.pyfunc.load_model(f"models:/{nombre}/{mv.version}")
                np.testing.assert_allclose(
                    recargado.predict(ejemplo), modelo.predict(ejemplo), rtol=1e-12, atol=1e-12
                )
                del X, modelo, recargado
        mlflow.log_input(
            mlflow.data.from_pandas(
                evaluacion[COLUMNAS],
                source=str(DATA_MANIFEST_PATH),
                digest=huella(evaluacion)[:32],
                targets="salidas",
                name="P1-holdout-octubre-15-31",
            ),
            context="evaluation",
        )
        resultados, diario = comparar_registrados(nombre, versiones, evaluacion)
        decision = decidir(resultados["A"], resultados["B"])
        for brazo, valores in resultados.items():
            valores["run_id"] = corridas[brazo]
            for metrica in ("mae", "rmse"):
                cliente.log_metric(corridas[brazo], f"actual_holdout_{metrica}", valores[metrica])
        evidencia = {
            "fecha_utc": datetime.now(UTC).isoformat(),
            "modelo": nombre,
            "tracking_uri": mlflow.get_tracking_uri(),
            "comparacion_run_id": padre.info.run_id,
            "protocolo": protocolo,
            "trazabilidad": trazabilidad,
            "resultados": resultados,
            "metricas_diarias": diario,
            "decision": {**decision, "version_recomendada": versiones[decision["recomendacion"]]},
            "entorno": {
                "python": platform.python_version(),
                **{
                    p: version(p) for p in ("mlflow", "scikit-learn", "numpy", "pandas", "pyarrow")
                },
            },
            "limitaciones": [
                "Comparación offline, no asignación aleatoria de tráfico en producción.",
                "A se reconstruye; no se afirma identidad con el artefacto original del equipo.",
                "B incorpora julio-septiembre y octubre 1-14; el test histórico deja de ser holdout para B.",
                "Un solo holdout de 17 días no demuestra superioridad causal ni permanente.",
                "Misma cohorte original; estaciones nuevas quedan fuera. Ceros no prueban operación.",
                "No se modifica champion. La versión recomendada requiere revisión antes de promoción.",
            ],
        }
        mlflow.log_dict(evidencia, "resultados_ab.json")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(evidencia, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return evidencia


def ejecutar(destino, tracking_uri, desde_csv=False):
    manifest = (
        json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
        if desde_csv
        else verificar_datos()
    )
    hash_manifest = sha256_archivo(DATA_MANIFEST_PATH)
    documentado, corrida = _modelo_documentado(manifest)
    if desde_csv:
        historico, arrastres = historico_desde_csv(manifest)
    else:
        historico = preparar(guardar=False)
        arrastres = pd.read_parquet(
            PROCESSED_DATA_DIR / "viajes.parquet",
            columns=["fin_del_viaje", "destino_id"],
            filters=[("fin_del_viaje", ">=", INICIO_ACTUAL), ("fin_del_viaje", "<", FIN_ACTUAL)],
        )
    drift.descargar_mes("2026-10")
    nuevo, stats = drift.cargar_mes("2026-10")
    esperado = json.loads(
        (REPORTS_DIR / "drift/resumen_p1_2026-10.json").read_text(encoding="utf-8")
    )
    if stats["sha256"] != esperado["archivo_actual"]["sha256"]:
        raise ValueError("El CSV actual no coincide con el utilizado en el reporte de drift.")
    dias = set(nuevo.inicio_del_viaje.dt.strftime("%Y-%m-%d"))
    if len(dias) != 31:
        raise ValueError("Faltan días observados en octubre.")
    contexto = historico.loc[
        historico.periodo >= INICIO_ACTUAL - pd.Timedelta(hours=HORAS_CONTEXTO)
    ]
    actual, cobertura = construir_actual(nuevo, contexto, arrastres, INICIO_ACTUAL, FIN_ACTUAL)
    a, b, evaluacion = particionar(historico, actual)
    if len(a) != int(corrida["metricas"]["train_filas"]):
        raise ValueError("A no reproduce la población de entrenamiento documentada.")
    del historico, actual, nuevo, contexto, arrastres
    trazabilidad = {
        "modelo_original_documentado": documentado,
        "data_version_historica": manifest["data_version"],
        "sha256_manifiesto": hash_manifest,
        "preparacion": "csv_congelados_solo_p1" if desde_csv else "parquet_ingesta_completa",
        "csv_actual": stats,
        "cobertura": cobertura,
        "codigo_sha256": {
            f"mibici/{n}.py": sha256_archivo(PROJ_ROOT / "mibici" / f"{n}.py")
            for n in ("ab_testing", "config", "preparar", "features", "drift", "monitoreo_p1")
        },
        "entrenamiento": {
            brazo: {
                "filas": len(frame),
                "sha256": huella(frame),
                "inicio": str(frame.periodo.min()),
                "fin": str(frame.periodo.max()),
            }
            for brazo, frame in (("A", a), ("B", b))
        },
    }
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("mibici-equipo27-ab-p1")
    resultado = entrenar_registrar(
        a, b, evaluacion, corrida["hp"], documentado["name"], trazabilidad, destino
    )
    if not desde_csv:
        verificar_datos()
    if sha256_archivo(DATA_MANIFEST_PATH) != hash_manifest:
        raise ValueError("El manifiesto cambió durante la comparación.")
    print(json.dumps(resultado["resultados"], indent=2))
    print(json.dumps(resultado["decision"], indent=2))
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="Registry de trabajo: usar una base local nueva para la reconstrucción.",
    )
    parser.add_argument(
        "--desde-csv",
        action="store_true",
        help="Reconstruir solo P1 desde los CSV y conteos verificados del manifiesto.",
    )
    parser.add_argument(
        "--salida", type=Path, default=REPORTS_DIR / "ab_testing/resultados_p1.json"
    )
    args = parser.parse_args()
    if args.salida.exists():
        parser.error(
            "La evidencia ya existe; elegir otra salida para una ejecución independiente."
        )
    ejecutar(args.salida, args.tracking_uri, desde_csv=args.desde_csv)


if __name__ == "__main__":
    main()
