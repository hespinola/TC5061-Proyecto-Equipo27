"""Pruebas de la preparación de datos para el análisis de drift (sin Evidently)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from mibici import drift


def estaciones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "latitude": [20.0, 20.1, 20.2],
            "longitude": [-103.0, -103.1, -103.2],
            "dpcapacity": [10.0, 20.0, np.nan],
        }
    )


def viajes(ids=(1, 2, 3, 9), minutos=(5, 10, 0.5, 300)) -> pd.DataFrame:
    inicio = pd.to_datetime(
        ["2026-10-03 08:00", "2026-10-03 09:30", "2026-10-04 23:10", "2026-10-05 00:00"]
    )
    return pd.DataFrame(
        {
            "viaje_id": [1, 2, 3, 4],
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.to_timedelta(minutos, unit="m"),
            "origen_id": list(ids),
            "destino_id": [2, 3, 1, 1],
        }
    )


class ConstruirVariablesTests(unittest.TestCase):
    def test_columnas_y_valores(self):
        out = drift.construir_variables(viajes(), estaciones())
        self.assertEqual(list(out.columns), list(drift.VARIABLES))
        self.assertEqual(out["duracion_min"].tolist(), [5, 10, 0.5, 300])
        self.assertEqual(out["hora_inicio"].tolist(), ["8", "9", "23", "0"])
        # 2026-10-03 es sábado (5) y 2026-10-05 es lunes (0).
        self.assertEqual(out["es_fin_de_semana"].tolist(), ["True", "True", "True", "False"])
        self.assertEqual(out["capacidad_origen"].iloc[:2].tolist(), [10.0, 20.0])
        self.assertTrue(np.isnan(out["capacidad_origen"].iloc[2]))  # capacidad inválida
        self.assertTrue(np.isnan(out["latitud_origen"].iloc[3]))  # estación 9 sin nomenclatura

    def test_no_incluye_campos_que_siempre_driftean_ni_personales(self):
        out = drift.construir_variables(viajes(), estaciones())
        for prohibida in ("viaje_id", "inicio_del_viaje", "usuario_id", "genero"):
            self.assertNotIn(prohibida, out.columns)


class CalidadTests(unittest.TestCase):
    def test_estaciones_nuevas_y_sin_nomenclatura(self):
        ref = drift.construir_variables(viajes(ids=(1, 2, 3, 1)), estaciones())
        act = drift.construir_variables(viajes(), estaciones())
        tabla = drift.resumen_calidad(ref, act, estaciones())
        self.assertEqual(tabla.loc["origen_sin_nomenclatura_pct", "referencia"], 0)
        self.assertEqual(tabla.loc["origen_sin_nomenclatura_pct", "actual"], 25)
        self.assertEqual(tabla.loc["estaciones_nuevas_en_actual", "actual"], 1)
        self.assertEqual(tabla.loc["estaciones_ausentes_en_actual", "actual"], 0)
        self.assertEqual(tabla.loc["duracion_mayor_240_min_pct", "actual"], 25)
        self.assertEqual(tabla.loc["duracion_menor_1_min_pct", "actual"], 25)

    def test_tabla_descartes_agrega_porcentajes(self):
        ref = pd.DataFrame(
            {
                "originales": [100],
                "invalidos": [0],
                "fuera_periodo": [0],
                "duplicados": [0],
                "validos": [100],
            },
            index=["2026-09"],
        )
        nuevo = {
            "originales": 200,
            "invalidos": 2,
            "fuera_periodo": 0,
            "duplicados": 4,
            "validos": 194,
        }
        tabla = drift.tabla_descartes(ref, nuevo, "2026-10")
        self.assertEqual(tabla.loc["2026-10", "invalidos_pct"], 1.0)
        self.assertEqual(tabla.loc["2026-10", "duplicados_pct"], 2.0)
        self.assertEqual(tabla.loc["2026-09", "invalidos_pct"], 0.0)


class CargarMesTests(unittest.TestCase):
    def test_normaliza_con_reglas_de_dataset_y_no_lee_campos_personales(self):
        csv = (
            "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,"
            "Origen_Id,Destino_Id\n"
            "1,10,F,1990,2026-10-01 08:00:00,2026-10-01 08:10:00,1,2\n"
            "1,10,F,1990,2026-10-01 08:00:00,2026-10-01 08:10:00,1,2\n"  # duplicado
            "2,11,M,1985,2026-10-02 09:00:00,2026-10-02 08:00:00,1,2\n"  # fin < inicio
            "3,12,M,1985,2026-09-30 23:00:00,2026-09-30 23:30:00,1,2\n"  # fuera del mes
            "4,13,F,2000,2026-10-31 23:50:00,2026-11-01 00:10:00,2,3\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "datos_abiertos_2026_10.csv").write_text(csv, encoding="utf-8")
            df, stats = drift.cargar_mes("2026-10", raw_dir=raw)
        self.assertEqual(df["viaje_id"].tolist(), [1, 4])
        self.assertEqual(
            (stats["originales"], stats["invalidos"], stats["fuera_periodo"], stats["duplicados"]),
            (5, 1, 1, 1),
        )
        self.assertEqual(stats["validos"], 2)
        self.assertNotIn("usuario_id", df.columns)
        self.assertNotIn("genero", df.columns)


class MuestreoYEvidentlyTests(unittest.TestCase):
    def test_muestreo_reproducible_y_sin_recorte_si_cabe(self):
        df = pd.DataFrame({"x": range(1000)})
        a, b = drift.muestrear(df, 100), drift.muestrear(df, 100)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 100)
        self.assertEqual(len(drift.muestrear(df, None)), 1000)
        self.assertEqual(len(drift.muestrear(df, 5000)), 1000)

    def test_tabla_drift_lee_veredicto_de_evidently(self):
        def metrica(col, metodo, valor):
            return {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": col,
                    "method": metodo,
                    "threshold": 0.1,
                },
                "value": valor,
            }

        resultado = {
            "metrics": [
                {"config": {"type": "evidently:metric_v2:RowCount"}, "value": 10.0},
                metrica("duracion_min", "Wasserstein distance (normed)", 0.5),
                metrica("origen_id", "Jensen-Shannon distance", 0.02),
            ],
            "tests": [
                {"name": "Value Drift for column duracion_min", "status": "TestStatus.FAIL"},
                {"name": "Value Drift for column origen_id", "status": "TestStatus.SUCCESS"},
                {"name": "Row count in dataset", "status": "TestStatus.SUCCESS"},
            ],
        }
        tabla = drift.tabla_drift(resultado)
        self.assertEqual(list(tabla.index), ["duracion_min", "origen_id"])
        self.assertEqual(tabla["drift"].tolist(), [True, False])
        self.assertEqual(tabla["tipo"].tolist(), ["numérica", "categórica"])


if __name__ == "__main__":
    unittest.main()