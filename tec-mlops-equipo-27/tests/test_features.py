"""Regresiones de calidad de datos y ausencia de fuga temporal en features.py.

Desde la raíz del proyecto generado:
    python -m unittest discover -s tests -v
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from mibici import features as f


def viajes_sinteticos(horas=200):
    inicio = pd.date_range("2026-06-24", periods=horas, freq="h")
    return pd.DataFrame(
        {
            f.COL_INICIO: inicio + pd.Timedelta(minutes=10),
            f.COL_FIN: inicio + pd.Timedelta(minutes=20),
            f.COL_ORIGEN: np.where(np.arange(horas) % 2 == 0, 1, 2),
            f.COL_DESTINO: np.where(np.arange(horas) % 2 == 0, 2, 1),
        }
    )


def nomenclatura_sintetica():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "dpcapacity": [20, 30],
            "latitude": [20.67, 20.68],
            "longitude": [-103.35, -103.36],
        }
    )


def panel_sintetico(horas=8, inicio="2026-06-30 20:00"):
    return pd.DataFrame(
        {
            f.COL_ESTACION: [1] * horas,
            f.COL_PERIODO: pd.date_range(inicio, periods=horas, freq="h"),
            "salidas": np.arange(horas),
            "llegadas": np.arange(horas) + 1,
            "flujo_neto": [1] * horas,
            "capacidad": [20] * horas,
        }
    )


class PanelTests(unittest.TestCase):
    def test_horas_sin_viajes_y_conservacion_de_totales(self):
        viajes = viajes_sinteticos(5).iloc[[0, 4]]
        panel = f.construir_panel_estacion_periodo(viajes)
        self.assertEqual(len(panel), 10)
        self.assertEqual(panel.salidas.sum(), 2)
        self.assertEqual(panel.llegadas.sum(), 2)
        self.assertEqual(panel.loc[panel.periodo.dt.hour == 2, "salidas"].sum(), 0)

    def test_conserva_estaciones_fuera_de_nomenclatura(self):
        viajes = viajes_sinteticos(8)
        panel = f.construir_features(viajes, nomenclatura_sintetica().iloc[[0]])
        self.assertEqual(set(panel.estacion_id), {1, 2})
        self.assertEqual(panel.salidas.sum(), len(viajes))
        self.assertEqual(panel.llegadas.sum(), len(viajes))
        faltantes = panel.loc[panel.estacion_id == 2]
        self.assertTrue(faltantes.capacidad.isna().all())
        self.assertTrue(faltantes.objetivo_en_riesgo.isna().all())

    def test_rechaza_viajes_vacios_fechas_invalidas_y_ids_nulos(self):
        casos = [viajes_sinteticos(0)]
        texto = viajes_sinteticos(2)
        texto[f.COL_INICIO] = texto[f.COL_INICIO].astype(str)
        casos.append(texto)
        nulos = viajes_sinteticos(2)
        nulos.loc[0, f.COL_ORIGEN] = np.nan
        casos.append(nulos)
        invertido = viajes_sinteticos(2)
        invertido[f.COL_FIN] = invertido[f.COL_INICIO] - pd.Timedelta(minutes=1)
        casos.append(invertido)
        for viajes in casos:
            with self.subTest(viajes=viajes), self.assertRaises((ValueError, TypeError)):
                f.construir_panel_estacion_periodo(viajes)


class TemporalTests(unittest.TestCase):
    def test_rezagos_y_medias_no_mezclan_estaciones(self):
        a = panel_sintetico(8)
        b = a.copy()
        b[f.COL_ESTACION] = 2
        b["salidas"] += 100
        panel = pd.concat([a, b])  # Índices repetidos: las funciones deben normalizarlos.
        panel = f.agregar_rezagos(panel, columnas=["salidas"], rezagos=[2])
        panel = f.agregar_promedios_moviles(panel, columnas=["salidas"], ventanas=[2])
        for estacion, base in ((1, 0), (2, 100)):
            grupo = panel.loc[panel.estacion_id == estacion]
            self.assertTrue(pd.isna(grupo.iloc[0].salidas_rezago_2))
            self.assertEqual(grupo.iloc[2].salidas_rezago_2, base)
            self.assertEqual(grupo.iloc[2].salidas_media_2, base + 0.5)

    def test_predictores_no_cambian_si_cambia_el_presente_o_futuro(self):
        viajes = viajes_sinteticos()
        t = pd.Timestamp("2026-07-01 03:00")
        extra = viajes.loc[viajes[f.COL_INICIO] >= t]
        original = f.construir_features(viajes, nomenclatura_sintetica())
        cambiado = f.construir_features(
            pd.concat([viajes, extra], ignore_index=True), nomenclatura_sintetica()
        )
        predictores = [
            c for c in original if "_rezago_" in c or "_media_" in c or "acumulado" in c
        ]
        assert_frame_equal(
            original.loc[original.periodo == t, predictores].reset_index(drop=True),
            cambiado.loc[cambiado.periodo == t, predictores].reset_index(drop=True),
        )

    def test_acumulado_excluye_hora_actual_y_reinicia_en_medianoche(self):
        panel = panel_sintetico()
        panel["flujo_neto"] = [1, 2, 3, 4, 100, 6, 7, 8]
        out = f.agregar_flujo_acumulado_dia(panel)
        self.assertEqual(out.flujo_acumulado_dia.tolist(), [0, 1, 3, 6, 0, 100, 106, 113])

    def test_wrapper_rechaza_anticipacion_no_soportada(self):
        for horizonte in (0, -1, 2, 3, True, 1.5):
            with self.subTest(horizonte=horizonte), self.assertRaises(ValueError):
                f.construir_features(
                    viajes_sinteticos(), nomenclatura_sintetica(), horizonte=horizonte
                )

    def test_rezagos_ventanas_y_horizontes_deben_ser_positivos(self):
        for fn, kwargs in (
            (f.agregar_rezagos, {"rezagos": []}),
            (f.agregar_rezagos, {"rezagos": [0]}),
            (f.agregar_rezagos, {"rezagos": [1], "horizonte": 2}),
            (f.agregar_promedios_moviles, {"ventanas": []}),
            (f.agregar_promedios_moviles, {"ventanas": [-1]}),
            (f.agregar_promedios_moviles, {"horizonte": 0}),
            (f.construir_objetivo_riesgo, {"horizonte": 0}),
            (f.construir_objetivo_riesgo, {"umbral": 0}),
            (f.construir_objetivo_riesgo, {"umbral": np.inf}),
        ):
            with self.subTest(fn=fn.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                fn(panel_sintetico(), **kwargs)

    def test_rechaza_huecos_y_claves_duplicadas(self):
        for panel in (
            panel_sintetico().drop(index=2),
            pd.concat([panel_sintetico(), panel_sintetico().iloc[[0]]]),
        ):
            for fn in (
                f.agregar_rezagos,
                f.agregar_promedios_moviles,
                f.construir_objetivo_riesgo,
            ):
                with self.subTest(fn=fn.__name__), self.assertRaises(ValueError):
                    fn(panel)

    def test_purga_ventanas_que_cruzan_corte_sin_perder_contexto(self):
        panel = f.construir_panel_estacion_periodo(viajes_sinteticos())
        panel = f.agregar_rezagos(panel)
        panel = f.agregar_capacidad(panel, nomenclatura_sintetica())
        panel = f.construir_objetivo_riesgo(panel)
        train = f.filtrar_periodo(panel, fin="2026-06-30", col_fin_objetivo="objetivo_fin_periodo")
        self.assertEqual(train.periodo.max(), pd.Timestamp("2026-06-30 21:00"))
        val = f.filtrar_periodo(
            panel,
            inicio="2026-07-01",
            fin="2026-07-31",
            col_fin_objetivo="objetivo_fin_periodo",
        )
        self.assertTrue(
            val.loc[val.periodo == pd.Timestamp("2026-07-01"), "salidas_rezago_168"].notna().all()
        )
        self.assertTrue(val.objetivo_en_riesgo.notna().all())
        # P1 conserva las últimas horas; no requiere la ventana futura de P2.
        p1 = f.filtrar_periodo(panel, fin="2026-06-30")
        self.assertEqual(p1.periodo.max(), pd.Timestamp("2026-06-30 23:00"))

    def test_purga_requiere_corte_y_respeta_objetivos_de_cada_particion(self):
        panel = panel_sintetico(12)
        a = f.construir_objetivo_riesgo(panel)
        modificado = panel.copy()
        modificado.loc[modificado.periodo >= "2026-07-01", "flujo_neto"] = 100
        b = f.construir_objetivo_riesgo(modificado)
        train_a = f.filtrar_periodo(a, fin="2026-06-30", col_fin_objetivo="objetivo_fin_periodo")
        train_b = f.filtrar_periodo(b, fin="2026-06-30", col_fin_objetivo="objetivo_fin_periodo")
        assert_series_equal(train_a.objetivo_en_riesgo, train_b.objetivo_en_riesgo)
        with self.assertRaises(ValueError):
            f.filtrar_periodo(a, col_fin_objetivo="objetivo_fin_periodo")


class CapacidadYObjetivoTests(unittest.TestCase):
    def test_union_no_multiplica_filas_y_rechaza_ids_duplicados(self):
        panel = panel_sintetico()
        nom = nomenclatura_sintetica()
        self.assertEqual(
            len(f.agregar_capacidad(panel.drop(columns="capacidad"), nom)), len(panel)
        )
        duplicada = pd.concat([nom, nom.iloc[[0]]])
        with self.assertRaises(ValueError):
            f.agregar_capacidad(panel, duplicada)

    def test_capacidad_invalida_no_produce_infinitos_ni_etiquetas(self):
        for capacidad in (None, 0, -1, np.inf, "invalida"):
            with self.subTest(capacidad=capacidad):
                panel = panel_sintetico()
                panel["capacidad"] = capacidad
                out = f.agregar_flujo_acumulado_dia(panel)
                self.assertTrue(out.flujo_acumulado_dia_rel.isna().all())
                out = f.construir_objetivo_riesgo(out)
                self.assertTrue(out.objetivo_en_riesgo.isna().all())
                self.assertTrue(out.objetivo_riesgo_sin_bicis.isna().all())
                self.assertTrue(out.objetivo_riesgo_sin_anclajes.isna().all())

    def test_umbral_inclusivo_direcciones_y_ventanas_incompletas(self):
        panel = panel_sintetico(6)
        panel["flujo_neto"] = [-3, -4, -3, 3, 4, 3]
        out = f.construir_objetivo_riesgo(panel)
        self.assertEqual(out.iloc[0].objetivo_riesgo_sin_bicis, 1)
        self.assertEqual(out.iloc[3].objetivo_riesgo_sin_anclajes, 1)
        self.assertEqual(out.iloc[1].objetivo_en_riesgo, 0)
        self.assertTrue(out.objetivo_en_riesgo.iloc[-2:].isna().all())
        self.assertTrue(out.objetivo_fin_periodo.iloc[-2:].isna().all())


class PerfilesYCLITests(unittest.TestCase):
    def test_perfil_mantiene_48_horas_si_faltan_dias_de_un_tipo(self):
        panel = f.agregar_variables_calendario(
            f.construir_panel_estacion_periodo(viajes_sinteticos(24))
        )
        out = f.construir_perfil_estaciones(panel, nomenclatura_sintetica())
        cols = [c for c in out if c.startswith("perfil_")]
        self.assertEqual(len(cols), 48)
        self.assertTrue((out.filter(like="perfil_fds_") == 0).all().all())
        self.assertTrue(np.allclose(out.filter(like="perfil_lab_").sum(axis=1), 1))
        duplicada = pd.concat([nomenclatura_sintetica(), nomenclatura_sintetica().iloc[[0]]])
        with self.assertRaises(ValueError):
            f.construir_perfil_estaciones(panel, duplicada)

    def test_cli_lee_y_escribe_parquet_creando_directorio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            viajes = tmp / "viajes.parquet"
            nom = tmp / "nomenclatura.parquet"
            salida = tmp / "nuevo" / "panel.parquet"
            viajes_sinteticos().to_parquet(viajes, index=False)
            nomenclatura_sintetica().to_parquet(nom, index=False)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mibici.features",
                    "--viajes-path",
                    str(viajes),
                    "--nomenclatura-path",
                    str(nom),
                    "--output-path",
                    str(salida),
                ],
                cwd=Path(f.__file__).resolve().parents[1],
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            panel = pd.read_parquet(salida)
            self.assertEqual(panel.salidas.sum(), 200)
            self.assertEqual(panel.llegadas.sum(), 200)
            self.assertIn("objetivo_fin_periodo", panel.columns)


if __name__ == "__main__":
    unittest.main()
