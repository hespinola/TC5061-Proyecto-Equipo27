"""Contratos que evitan fuga, cambio de baseline y pérdida del umbral al servir."""

import copy
import json
import unittest

import cloudpickle
import numpy as np
import pandas as pd

from mibici.config import BUSQUEDAS, FEATURES_P1, FEATURES_P2, REPORTS_DIR
from mibici.modeling.candidates import cluster_scores, elegir_umbral
from mibici.modeling.estimators import FrequencyClassifier, ThresholdClassifier
from mibici.modeling.experiment import comprobar_referencia


class ExperimentTests(unittest.TestCase):
    def test_busquedas_tienen_tres_combinaciones_y_features_sin_futuro(self):
        self.assertTrue(all(len(v) >= 3 for v in BUSQUEDAS.values()))
        forbidden = {"salidas", "llegadas", "flujo_neto", "objetivo_en_riesgo",
                     "objetivo_flujo_futuro", "objetivo_fin_periodo", "viaje_id", "usuario_id"}
        self.assertFalse(forbidden.intersection(FEATURES_P1 + FEATURES_P2))
        self.assertFalse(any(c.startswith("objetivo_") for c in FEATURES_P1 + FEATURES_P2))

    def test_umbral_se_conserva_tras_serializar_y_cambia_las_alertas(self):
        X = pd.DataFrame({"estacion_id": [1, 1, 1, 2], "hora_del_dia": [8, 8, 8, 8]})
        base = FrequencyClassifier().fit(X, [1, 0, 0, 0])
        model = ThresholdClassifier(base, 0.3)
        restored = cloudpickle.loads(cloudpickle.dumps(model))
        self.assertEqual(restored.predict(X).tolist(), [1, 1, 1, 0])
        self.assertTrue(np.array_equal(model.predict_proba(X), restored.predict_proba(X)))
        self.assertEqual(base.predict(X).sum(), 0)

    def test_umbral_se_elige_solo_con_validacion(self):
        threshold, table = elegir_umbral([0, 0, 1, 1], [0.01, 0.04, 0.15, 0.9])
        self.assertEqual(threshold, 0.1)
        self.assertEqual(max(row["f1"] for row in table), 1)

    def test_baseline_rechaza_metricas_o_version_distintas(self):
        ref = json.loads((REPORTS_DIR / "baseline_results.json").read_text())
        rows = [dict(r, stage_of_work="baseline") for r in ref["runs"]]
        self.assertEqual(len(comprobar_referencia(rows, ref["data_version"])), 4)
        bad = copy.deepcopy(rows)
        bad[0]["metricas"]["test_mae"] += 0.01
        with self.assertRaises(ValueError):
            comprobar_referencia(bad, ref["data_version"])
        with self.assertRaises(ValueError):
            comprobar_referencia(rows, "otra_version")

    def test_clusters_colapsados_no_parecen_buenos(self):
        score = cluster_scores(np.arange(12).reshape(4, 3), [0, 0, 0, 0])
        self.assertEqual(score["collapsed"], 1)
        self.assertEqual(score["silhouette"], -1)


if __name__ == "__main__":
    unittest.main()
