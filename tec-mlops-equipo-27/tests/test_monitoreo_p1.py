"""Contratos temporales, cobertura y preservación del monitoreo de P1."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mibici import drift
from mibici.config import FEATURES_P1, FIN_ENTRENAMIENTO
from mibici.dataset import sha256_archivo, version_manifest
from mibici.monitoreo_p1 import HORAS_CONTEXTO, construir_actual, crear_dataset_p1
from mibici.preparar import preparar


class ActualP1Tests(unittest.TestCase):
    def setUp(self):
        self.inicio = pd.Timestamp("2026-10-01")
        self.fin = pd.Timestamp("2026-11-01")
        horas = pd.date_range(
            self.inicio - pd.Timedelta(hours=HORAS_CONTEXTO),
            self.inicio,
            freq="h",
            inclusive="left",
        )
        self.contexto = pd.MultiIndex.from_product(
            [[1, 2], horas], names=["estacion_id", "periodo"]
        ).to_frame(index=False)
        self.contexto["salidas"] = 1
        self.contexto["llegadas"] = 2
        self.nuevo = pd.DataFrame(
            {
                "viaje_id": [100],
                "inicio_del_viaje": pd.to_datetime(["2026-10-01 00:00"]),
                "fin_del_viaje": pd.to_datetime(["2026-10-01 00:10"]),
                "origen_id": [1],
                "destino_id": [1],
            }
        )
        self.arrastres = pd.DataFrame(
            {
                "fin_del_viaje": pd.to_datetime(["2026-10-01 00:05"]),
                "destino_id": [1],
            }
        )

    def construir(self, nuevo=None, contexto=None):
        return construir_actual(
            self.nuevo if nuevo is None else nuevo,
            self.contexto if contexto is None else contexto,
            self.arrastres,
            self.inicio,
            self.fin,
        )

    def test_conserva_horas_sin_viajes_y_llegadas_que_cruzan_el_corte(self):
        actual, cobertura = self.construir()
        self.assertEqual(len(actual), 2 * 31 * 24)
        self.assertFalse(actual.duplicated(["estacion_id", "periodo"]).any())
        self.assertEqual(int(actual.salidas.sum()), 1)
        self.assertEqual(int(actual.llegadas.sum()), 2)
        primera = actual.loc[
            (actual.estacion_id == 1) & (actual.periodo == self.inicio)
        ].iloc[0]
        self.assertEqual(primera.salidas_rezago_24, 1)
        self.assertEqual(primera.salidas_rezago_168, 1)
        self.assertEqual(primera.salidas_media_24, 1)
        self.assertEqual(primera.llegadas_media_24, 2)
        self.assertEqual(primera.llegadas, 2)
        self.assertFalse(actual[list(FEATURES_P1)].isna().any().any())
        self.assertEqual(cobertura["llegadas_historicas_en_mes"], 1)

    def test_un_viaje_no_modifica_predictores_hasta_el_inicio_de_su_hora(self):
        futuro = self.nuevo.copy()
        futuro["viaje_id"] = 101
        futuro["inicio_del_viaje"] = pd.Timestamp("2026-10-02 12:00")
        futuro["fin_del_viaje"] = pd.Timestamp("2026-10-02 12:10")
        original, _ = self.construir()
        modificado, _ = self.construir(pd.concat([self.nuevo, futuro], ignore_index=True))
        antes = original.periodo <= pd.Timestamp("2026-10-02 12:00")
        pd.testing.assert_frame_equal(
            original.loc[antes, list(FEATURES_P1)].reset_index(drop=True),
            modificado.loc[antes, list(FEATURES_P1)].reset_index(drop=True),
        )
        hora_modificada = (
            (original.estacion_id == 1)
            & (original.periodo == pd.Timestamp("2026-10-02 12:00"))
        )
        self.assertEqual(
            int(modificado.loc[hora_modificada, "salidas"].sum()),
            int(original.loc[hora_modificada, "salidas"].sum()) + 1,
        )

    def test_estaciones_nuevas_se_reportan_sin_inventarles_historia(self):
        nuevo = self.nuevo.copy()
        nuevo["origen_id"] = 99
        actual, cobertura = self.construir(nuevo)
        self.assertEqual(set(actual.estacion_id), {1, 2})
        self.assertEqual(cobertura["estaciones_nuevas_fuera_cohorte"], [99])
        self.assertEqual(cobertura["salidas_fuera_cohorte"], 1)
        self.assertEqual(int(actual.llegadas.sum()), 2)

    def test_rechaza_contexto_incompleto(self):
        with self.assertRaisesRegex(ValueError, "Faltan horas"):
            self.construir(contexto=self.contexto.iloc[1:])


class PreservacionTests(unittest.TestCase):
    def preparar_archivos(self, root):
        viajes = pd.DataFrame(
            {
                "viaje_id": [1, 2, 3],
                "inicio_del_viaje": pd.to_datetime(
                    ["2026-09-01 00:00", "2026-09-08 00:00", "2026-09-30 23:55"]
                ),
                "fin_del_viaje": pd.to_datetime(
                    ["2026-09-01 00:10", "2026-09-08 00:10", "2026-10-01 00:10"]
                ),
                "origen_id": [1, 2, 1],
                "destino_id": [2, 1, 2],
            }
        )
        nomenclatura = pd.DataFrame({"id": [1, 2], "dpcapacity": [10.0, 20.0]})
        viajes.to_parquet(root / "viajes.parquet", index=False)
        nomenclatura.to_parquet(root / "nomenclatura.parquet", index=False)
        manifest = {
            "inicio": "2026-09-01",
            "fin_exclusivo": "2026-10-01",
            "resumen": {"validos": 3},
            "salidas": {
                nombre: {"sha256": sha256_archivo(root / nombre)}
                for nombre in ("viajes.parquet", "nomenclatura.parquet")
            },
        }
        manifest["data_version"] = version_manifest(manifest)
        path = root / "data_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_preparar_sin_guardar_preserva_los_archivos_congelados(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.preparar_archivos(root)
            archivos = [root / "viajes.parquet", root / "nomenclatura.parquet", manifest]
            anteriores = {path: sha256_archivo(path) for path in archivos}
            panel = preparar(root, manifest, guardar=False)
            self.assertFalse((root / "panel_estacion_hora.parquet").exists())
            self.assertEqual(int(panel.salidas.sum()), 3)
            self.assertEqual(int(panel.llegadas.sum()), 2)
            self.assertEqual(anteriores, {path: sha256_archivo(path) for path in archivos})

    def test_preparar_conserva_la_escritura_por_defecto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.preparar_archivos(root)
            preparar(root, manifest)
            self.assertTrue((root / "panel_estacion_hora.parquet").exists())


class ContratosDriftTests(unittest.TestCase):
    def test_referencia_por_defecto_excluye_validacion_y_prueba(self):
        corte = pd.Timestamp(FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)
        datos = pd.DataFrame(
            {
                "inicio_del_viaje": [
                    corte - pd.Timedelta(hours=1),
                    corte,
                    corte + pd.Timedelta(hours=1),
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datos.to_parquet(root / "viajes.parquet", index=False)
            referencia = drift.cargar_referencia(processed_dir=root)
            self.assertEqual(referencia.inicio_del_viaje.tolist(), [datos.iloc[0, 0]])
            explicita = drift.cargar_referencia(
                meses=[corte.strftime("%Y-%m")], processed_dir=root
            )
            self.assertEqual(len(explicita), 2)

    def test_un_veredicto_ausente_no_se_interpreta_como_ausencia_de_drift(self):
        resultado = {
            "metrics": [
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "salidas_rezago_24",
                        "method": "wasserstein",
                        "threshold": 0.1,
                    },
                    "value": 0.8,
                }
            ],
            "tests": [],
        }
        with self.assertRaisesRegex(ValueError, "veredicto"):
            drift.tabla_drift(resultado, numericas=("salidas_rezago_24",))

    def test_cargar_mes_selecciona_columnas_operativas_al_leer(self):
        csv = (
            "Viaje_Id,Usuario_Id,Genero,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id\n"
            "1,99,F,2026-10-01 08:00,2026-10-01 08:10,1,2\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datos_abiertos_2026_10.csv").write_text(csv, encoding="utf-8")
            with patch.object(drift.pd, "read_csv", wraps=pd.read_csv) as lector:
                _, stats = drift.cargar_mes("2026-10", raw_dir=root)
            selector = lector.call_args.kwargs.get("usecols")
            self.assertTrue(callable(selector))
            self.assertTrue(selector("Inicio_del_viaje"))
            self.assertTrue(selector("Id_Viaje"))
            self.assertFalse(selector("Usuario_Id"))
            self.assertFalse(selector("Genero"))
            self.assertIn("sha256", stats)

    def test_cargar_mes_rechaza_duplicados_con_datos_contradictorios(self):
        csv = (
            "Viaje_Id,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id\n"
            "1,2026-10-01 08:00,2026-10-01 08:10,1,2\n"
            "1,2026-10-01 08:00,2026-10-01 08:10,1,3\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datos_abiertos_2026_10.csv").write_text(csv, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contradictorios"):
                drift.cargar_mes("2026-10", raw_dir=root)


class EvidentlyIntegracionTests(unittest.TestCase):
    def test_calidad_acepta_calendario_int8_sin_modificar_entradas_del_modelo(self):
        from evidently import Report
        from evidently.presets import DataSummaryPreset

        frame = pd.DataFrame(
            {columna: [float(i % 7) for i in range(48)] for columna in FEATURES_P1}
        )
        for columna, valores in (
            ("hora_del_dia", list(range(24)) * 2),
            ("dia_semana", [i % 7 for i in range(48)]),
            ("es_fin_de_semana", [int(i % 7 >= 5) for i in range(48)]),
            ("mes", [10] * 48),
        ):
            frame[columna] = pd.Series(valores, dtype="int8")
        original = frame.copy(deep=True)
        dataset = crear_dataset_p1(frame)
        snapshot = Report([DataSummaryPreset()]).run(dataset, dataset)
        self.assertTrue(snapshot.dict()["metrics"])
        pd.testing.assert_frame_equal(frame, original)

    def test_snapshot_real_distingue_cambio_y_distribucion_identica(self):
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        referencia = pd.DataFrame(
            {
                "salidas_rezago_24": [float(x) for x in range(2000)],
                "dia_semana": ["laboral", "fin_de_semana"] * 1000,
            }
        )
        actual = referencia.copy()
        actual["salidas_rezago_24"] += 100_000
        definicion = DataDefinition(
            numerical_columns=["salidas_rezago_24"],
            categorical_columns=["dia_semana"],
        )
        ds_ref = Dataset.from_pandas(referencia, data_definition=definicion)
        ds_act = Dataset.from_pandas(actual, data_definition=definicion)
        snapshot = Report(
            [
                DataDriftPreset(
                    num_method="wasserstein",
                    cat_method="jensenshannon",
                    threshold=0.1,
                )
            ],
            include_tests=True,
        ).run(ds_act, ds_ref)
        tabla = drift.tabla_drift(snapshot.dict(), numericas=("salidas_rezago_24",))
        self.assertTrue(tabla.loc["salidas_rezago_24", "drift"])
        self.assertFalse(tabla.loc["dia_semana", "drift"])
        self.assertFalse(tabla.drift.isna().any())
        self.assertFalse(tabla.metodo.isna().any())


class MonitoreoDesdeCsvTests(unittest.TestCase):
    """La ruta --desde-csv no toca Parquet congelados y produce los mismos artefactos."""

    def _historico(self):
        from mibici import features as f

        rng = np.random.default_rng(0)
        horas = pd.date_range("2025-10-01", "2026-10-01", freq="h", inclusive="left")
        panel = pd.MultiIndex.from_product(
            [[1, 2], horas], names=["estacion_id", "periodo"]
        ).to_frame(index=False)
        panel["salidas"] = rng.poisson(2, len(panel))
        panel["llegadas"] = rng.poisson(2, len(panel))
        panel = f.agregar_variables_calendario(panel)
        panel = f.agregar_rezagos(panel)
        panel = f.agregar_promedios_moviles(panel)
        arrastres = pd.DataFrame(
            {
                "fin_del_viaje": pd.to_datetime(["2026-10-01 00:03"]),
                "destino_id": [2],
            }
        )
        return panel, arrastres

    def _octubre(self):
        dias = pd.date_range("2026-10-01 08:00", periods=31, freq="D")
        nuevo = pd.DataFrame(
            {
                "viaje_id": range(1, 63),
                "inicio_del_viaje": list(dias) * 2,
                "fin_del_viaje": [d + pd.Timedelta(minutes=10) for d in dias] * 2,
                "origen_id": [1] * 31 + [2] * 31,
                "destino_id": [2] * 31 + [1] * 31,
            }
        )
        stats = {"validos": len(nuevo), "sha256": "x", "archivo": "datos_abiertos_2026_10.csv"}
        return nuevo, stats

    def test_desde_csv_genera_reportes_sin_parquet_ni_verificacion_congelada(self):
        from mibici import monitoreo_p1 as m

        historico, arrastres = self._historico()
        corte = pd.Timestamp(FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)
        filas_ref = int(
            historico.loc[historico.periodo < corte]
            .dropna(subset=[*FEATURES_P1, "salidas"])
            .shape[0]
        )
        modelo = {
            "name": "modelo",
            "version": "1",
            "run_id": "r",
            "source_commit": "c",
            "data_version": "d",
        }
        corrida = {"metricas": {"train_filas": filas_ref}, "hp": {"max_depth": 8}}
        prohibido = AssertionError("La ruta CSV no debe usar los Parquet congelados.")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "mibici.ab_testing.historico_desde_csv", return_value=(historico, arrastres)
            ) as reconstruir,
            patch.object(m, "_modelo_documentado", return_value=(modelo, corrida)),
            patch.object(m.drift, "descargar_mes"),
            patch.object(m.drift, "cargar_mes", return_value=self._octubre()),
            patch.object(m, "verificar_datos", side_effect=prohibido),
            patch.object(m, "preparar", side_effect=prohibido),
            patch.object(m.pd, "read_parquet", side_effect=prohibido),
        ):
            resumen = m.ejecutar(muestra=None, desde_csv=True, destino=tmp)
            reconstruir.assert_called_once()
            salida = Path(tmp)
            for nombre in (
                "calidad_p1_2026-10_vs_entrenamiento.html",
                "drift_p1_2026-10_vs_entrenamiento.html",
                "drift_p1_2026-10_vs_2025-10.html",
                "drift_p1_2026-10.png",
                "resumen_p1_2026-10.json",
            ):
                self.assertTrue((salida / nombre).exists(), nombre)
        self.assertEqual(resumen["preparacion"], "csv_congelados_solo_p1")
        self.assertEqual(resumen["filas"]["referencia_elegibles"], filas_ref)
        self.assertEqual(resumen["filas"]["actual_elegibles"], 2 * 31 * 24)
        self.assertEqual(len(resumen["drift_por_variable"]), len(FEATURES_P1))


if __name__ == "__main__":
    unittest.main()
