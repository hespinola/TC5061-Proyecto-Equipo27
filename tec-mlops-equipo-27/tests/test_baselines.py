"""Comprobaciones de baselines y de sus límites temporales sin un servidor MLflow."""

import unittest

import pandas as pd

from mibici.modeling.train import frecuencias_riesgo, metricas_clasificacion, particiones


class BaselinesTests(unittest.TestCase):
    def test_frecuencias_no_aprenden_del_objetivo_de_evaluacion(self):
        train = pd.DataFrame(
            {"estacion_id": [1, 1, 2], "hora_del_dia": [8, 8, 9], "objetivo_en_riesgo": [0, 1, 1]}
        )
        test = pd.DataFrame(
            {"estacion_id": [1, 99], "hora_del_dia": [8, 8], "objetivo_en_riesgo": [1, 0]}
        )
        a = frecuencias_riesgo(train, test)
        test["objetivo_en_riesgo"] = 1 - test.objetivo_en_riesgo
        b = frecuencias_riesgo(train, test)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertAlmostEqual(a[0], 0.5)
        self.assertAlmostEqual(a[1], 2 / 3)

    def test_metricas_y_validacion_de_clases(self):
        m = metricas_clasificacion([0, 1, 1], [0.1, 0.8, 0.9], 0.5)
        self.assertEqual(m["f1"], 1)
        self.assertEqual(m["average_precision"], 1)
        with self.assertRaises(ValueError):
            metricas_clasificacion([0, 0], [0.1, 0.2], 0.5)

    def test_particiones_p2_no_cruzan_cortes(self):
        p = pd.DataFrame(
            {
                "periodo": pd.to_datetime(
                    [
                        "2026-06-30 21:00",
                        "2026-06-30 22:00",
                        "2026-07-31 21:00",
                        "2026-07-31 22:00",
                        "2026-09-30 21:00",
                        "2026-09-30 22:00",
                    ]
                ),
            }
        )
        p["objetivo_fin_periodo"] = p.periodo + pd.Timedelta(hours=3)
        partes = particiones(p, "2026-10-01", p2=True)
        self.assertEqual([len(partes[k]) for k in ("train", "validation", "test")], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
