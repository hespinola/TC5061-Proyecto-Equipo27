"""Cortes temporales, decisión conservadora y registro/comparación real en SQLite."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici import ab_testing as ab
from mibici import drift
from mibici.config import FEATURES_P1
from mibici.dataset import normalizar_estaciones, sha256_archivo, version_manifest
from mibici.preparar import preparar


def panel(inicio, fin):
    fechas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    frame = pd.DataFrame({"periodo": fechas, "estacion_id": 1})
    for columna in FEATURES_P1:
        frame[columna] = np.arange(len(frame), dtype=float) % 7
    frame["salidas"] = np.arange(len(frame), dtype=float) % 5
    return frame


@pytest.fixture
def partes():
    return panel("2026-06-28", "2026-10-01"), panel("2026-10-01", "2026-11-01")


def test_cortes_no_mezclan_entrenamiento_y_evaluacion(partes):
    a, b, evaluacion = ab.particionar(*partes)
    assert a.periodo.max() == pd.Timestamp("2026-06-30T23:00")
    assert b.periodo.max() == pd.Timestamp("2026-10-14T23:00")
    assert evaluacion.periodo.min() == pd.Timestamp("2026-10-15")
    assert len(evaluacion) == 17 * 24
    assert evaluacion.periodo.max() == pd.Timestamp("2026-10-31T23:00")
    assert set(b.periodo).isdisjoint(evaluacion.periodo)


def test_cambiar_objetivos_holdout_no_cambia_entrenamiento(partes):
    historico, actual = partes
    a, b, evaluacion = ab.particionar(historico, actual)
    alterado = actual.copy()
    alterado.loc[alterado.periodo >= ab.INICIO_EVALUACION, "salidas"] += 50
    a2, b2, evaluacion2 = ab.particionar(historico, alterado)
    pd.testing.assert_frame_equal(a, a2)
    pd.testing.assert_frame_equal(b, b2)
    assert ab.huella(evaluacion) != ab.huella(evaluacion2)


@pytest.mark.parametrize(
    "defecto", ["duplicado", "hora_ausente", "nan", "inf", "fuera", "estacion"]
)
def test_rechaza_evaluacion_incompleta_o_invalida(partes, defecto):
    historico, actual = partes
    if defecto == "duplicado":
        actual = pd.concat([actual, actual.head(1)])
    elif defecto == "hora_ausente":
        actual = actual.iloc[1:]
    elif defecto in ("nan", "inf"):
        actual.loc[0, FEATURES_P1[0]] = float(defecto)
    elif defecto == "fuera":
        actual.loc[0, "periodo"] = pd.Timestamp("2026-11-01")
    else:
        actual.loc[0, "estacion_id"] = 99
    with pytest.raises(ValueError):
        ab.particionar(historico, actual)


@pytest.mark.parametrize(
    "mae,rmse,ganador", [(0.98, 1.1, "B"), (0.995, 1.1, "A"), (1.0, 1.1, "A"), (0.8, 1.3, "A")]
)
def test_decision_exige_mejora_mae_y_no_empeorar_rmse(mae, rmse, ganador):
    resultado = ab.decidir({"mae": 1.0, "rmse": 1.2}, {"mae": mae, "rmse": rmse})
    assert resultado["recomendacion"] == ganador
    assert resultado["promocion_automatica"] is False


def test_baseline_perfecto_y_metricas_invalidas():
    assert ab.decidir({"mae": 0, "rmse": 0}, {"mae": 0, "rmse": 0})["recomendacion"] == "A"
    with pytest.raises(ValueError):
        ab.decidir({"mae": 1, "rmse": 1}, {"mae": np.nan, "rmse": 1})


def test_ruta_csv_reproduce_features_de_preparacion_original(tmp_path, monkeypatch):
    path = tmp_path / "datos_abiertos_2026_09.csv"
    path.write_text(
        "Viaje_Id,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id\n"
        "1,2026-09-01 00:00,2026-09-01 00:10,1,2\n"
        "2,2026-09-30 23:00,2026-10-01 00:10,2,1\n",
        encoding="utf-8",
    )
    viajes, stats = drift.cargar_mes("2026-09", raw_dir=tmp_path)
    viajes.to_parquet(tmp_path / "viajes.parquet", index=False)
    nom = normalizar_estaciones(
        pd.DataFrame(
            {
                "id": [1, 2],
                "latitude": [20.0, 20.1],
                "longitude": [-103.0, -103.1],
                "dpcapacity": [10, 20],
            }
        )
    )
    nom.to_parquet(tmp_path / "nomenclatura.parquet", index=False)
    manifest = {
        "inicio": "2026-09-01",
        "fin_exclusivo": "2026-10-01",
        "fuentes": {"2026-09": "https://example.invalid/datos.csv"},
        "archivos": {"2026-09": stats},
        "resumen": {"validos": 2, "estaciones_observadas": 2},
        "salidas": {
            n: {"sha256": sha256_archivo(tmp_path / n)}
            for n in ("viajes.parquet", "nomenclatura.parquet")
        },
    }
    manifest["data_version"] = version_manifest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def sin_red(*args, **kwargs):
        raise AssertionError("Los CSV ya verificados no deben descargarse de nuevo")

    monkeypatch.setattr(ab.requests.Session, "get", sin_red)
    nuevo, arrastres = ab.historico_desde_csv(manifest, raw_dir=tmp_path)
    original = preparar(processed_dir=tmp_path, manifest_path=manifest_path, guardar=False)
    pd.testing.assert_frame_equal(nuevo[ab.COLUMNAS], original[ab.COLUMNAS])
    assert len(arrastres) == 1
    assert arrastres.iloc[0].fin_del_viaje == pd.Timestamp("2026-10-01T00:10")
    assert not (tmp_path / "panel_estacion_hora.parquet").exists()

    manifest["resumen"]["validos"] += 1
    manifest["data_version"] = version_manifest(manifest)
    with pytest.raises(ValueError, match="conteo"):
        ab.historico_desde_csv(manifest, raw_dir=tmp_path)


def test_registra_dos_versiones_y_compara_mismas_filas(tmp_path, partes):
    anterior, registry_anterior = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experimento = mlflow.create_experiment(
        "ab-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experimento)
    try:
        a, b, evaluacion = ab.particionar(*partes)
        destino = tmp_path / "resultado.json"
        resultado = ab.entrenar_registrar(
            a,
            b,
            evaluacion,
            {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.08},
            "modelo-p1-test",
            {"datos": "sinteticos"},
            destino,
        )
        modelos = resultado["resultados"]
        assert modelos["A"]["version"] != modelos["B"]["version"]
        assert modelos["A"]["evaluacion_sha256"] == modelos["B"]["evaluacion_sha256"]
        assert modelos["A"]["filas"] == modelos["B"]["filas"] == len(evaluacion)
        assert len(resultado["metricas_diarias"]) == 17
        assert json.loads(destino.read_text(encoding="utf-8"))["decision"] == resultado["decision"]
        cliente = mlflow.MlflowClient()
        assert not cliente.get_registered_model("modelo-p1-test").aliases
        for brazo in ("A", "B"):
            mv = cliente.get_model_version("modelo-p1-test", modelos[brazo]["version"])
            assert mv.run_id == modelos[brazo]["run_id"]
            assert mv.tags["brazo"] == brazo
            assert "actual_holdout_mae" in cliente.get_run(mv.run_id).data.metrics
        with pytest.raises(FileExistsError):
            ab.entrenar_registrar(a, b, evaluacion, {}, "modelo-p1-test", {}, destino)
        with pytest.raises(ValueError, match="solapan"):
            ab.entrenar_registrar(
                a,
                pd.concat([b, evaluacion]),
                evaluacion,
                {},
                "modelo-p1-test",
                {},
                tmp_path / "otro.json",
            )
    finally:
        mlflow.set_tracking_uri(anterior)
        mlflow.set_registry_uri(registry_anterior)
