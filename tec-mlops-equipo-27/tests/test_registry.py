"""La selección ignora prueba y aplica reglas explícitas de estabilidad/desempate."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from mibici.modeling.registry import ejemplo_modelo, seleccionar


def row(question, name, value, **metrics):
    key = {"p1": "validation_mae", "p2": "validation_average_precision",
           "p3": "validation_silhouette"}[question]
    return {"pregunta": question.upper(), "run_id": name, "metricas": {
        key: value, "validation_weekly_mae_std": 0.01, "model_bytes": 100,
        "validation_recall": 0.4, "validation_f1": 0.5, "validation_collapsed": 0,
        "min_cluster_size": 10, "seed_ari_min": 0.9, "validation_temporal_ari": 0.7,
        **metrics}}


class RegistryTests(unittest.TestCase):
    def test_ejemplo_descarga_uri_completo_de_logged_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "input_example.json").write_text(json.dumps({"columns": ["x"], "data": [[1.0]]}))
            model = SimpleNamespace(signature=True, saved_input_example_info={"artifact_path": "input_example.json"})
            with patch("mlflow.models.Model.load", return_value=model), patch(
                "mlflow.artifacts.download_artifacts", return_value=tmp
            ) as download:
                example = ejemplo_modelo("models:/m-abc")
                download.assert_called_once_with("models:/m-abc")
            self.assertEqual(example.x.tolist(), [1.0])

    def test_prueba_no_cambia_modelo_elegido(self):
        for q in ("p1", "p2", "p3"):
            rows = [row(q, "a", 0.6), row(q, "b", 0.3)]
            before, _ = seleccionar(rows, q)
            modified = copy.deepcopy(rows)
            modified[0]["metricas"].update(test_mae=999, test_average_precision=0, test_silhouette=-1)
            modified[1]["metricas"].update(test_mae=0, test_average_precision=1, test_silhouette=1)
            after, criterion = seleccionar(modified, q)
            self.assertEqual(before["run_id"], after["run_id"])
            self.assertFalse(criterion["test_used_for_selection"])

    def test_p1_casi_empate_prefiere_estabilidad(self):
        rows = [row("p1", "a", 0.5, validation_weekly_mae_std=0.1),
                row("p1", "b", 0.502, validation_weekly_mae_std=0.01)]
        chosen, _ = seleccionar(rows, "p1")
        self.assertEqual(chosen["run_id"], "b")

    def test_p2_casi_empate_prefiere_recall(self):
        rows = [row("p2", "a", 0.5, validation_recall=0.1),
                row("p2", "b", 0.498, validation_recall=0.5)]
        self.assertEqual(seleccionar(rows, "p2")[0]["run_id"], "b")

    def test_p3_rechaza_colapso_y_considera_semillas(self):
        rows = [row("p3", "a", 0.12, validation_collapsed=1),
                row("p3", "b", 0.10, seed_ari_min=0.5),
                row("p3", "c", 0.098, seed_ari_min=0.9)]
        self.assertEqual(seleccionar(rows, "p3")[0]["run_id"], "c")
        with self.assertRaises(ValueError):
            seleccionar(rows[:1], "p3")


if __name__ == "__main__":
    unittest.main()
