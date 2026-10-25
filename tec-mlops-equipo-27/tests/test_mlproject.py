"""Contrato del MLflow Project: entorno, puntos de entrada y valores por defecto."""

import inspect
from pathlib import Path
import unittest

from mlflow.projects import _project_spec
import yaml

from mibici.config import HIPERPARAMETROS, PROJ_ROOT, REZAGOS_HORAS, VALOR_DESDE_CONFIG
from mibici.modeling import train

REPO_ROOT = PROJ_ROOT.parent


def requisitos_fijos(path: Path) -> set[str]:
    lineas = (linea.strip() for linea in path.read_text(encoding="utf-8").splitlines())
    return {x for x in lineas if x and not x.startswith(("#", "-e"))}


class MLprojectTests(unittest.TestCase):
    def test_entorno_usa_las_mismas_versiones_fijas(self):
        entorno = yaml.safe_load((REPO_ROOT / "python_env.yaml").read_text(encoding="utf-8"))
        self.assertEqual(entorno["python"], "3.13")
        self.assertEqual(
            set(entorno["dependencies"]), requisitos_fijos(PROJ_ROOT / "requirements.txt")
        )
        self.assertTrue(all("==" in d for d in entorno["dependencies"]))

    def test_puntos_de_entrada_y_parametros_de_train(self):
        proyecto = _project_spec.load_project(str(REPO_ROOT))
        self.assertEqual(proyecto.env_type, "python_env")
        self.assertEqual(set(proyecto._entry_points), {"ingest", "train"})
        parametros = proyecto.get_entry_point("train").parameters
        # Cada parámetro del MLproject es una opción de train.main y viceversa.
        self.assertEqual(set(parametros), set(inspect.signature(train.main).parameters))
        for nombre, parametro in parametros.items():
            esperado = "all" if nombre == "question" else VALOR_DESDE_CONFIG
            self.assertEqual(parametro.default, esperado, nombre)

    def test_comando_train_es_reproducible(self):
        proyecto = _project_spec.load_project(str(REPO_ROOT))
        comando = proyecto.get_entry_point("train").compute_command(
            {"question": "p1"}, storage_dir=None
        )
        self.assertIn("python -m mibici.modeling.train", comando)
        self.assertIn("--question p1", comando)
        self.assertIn(f"--model {VALOR_DESDE_CONFIG}", comando)


class ParametrosTrainTests(unittest.TestCase):
    def test_centinela_conserva_valor_de_config(self):
        hp = train.resolver_hiperparametros("p1", "baseline", {"rezago_horas": "config"})
        self.assertEqual(hp, HIPERPARAMETROS["p1"]["baseline"])
        self.assertIn(hp["rezago_horas"], REZAGOS_HORAS)

    def test_valores_recibidos_se_convierten(self):
        self.assertEqual(
            train.resolver_hiperparametros("p1", "baseline", {"rezago_horas": "24"}),
            {"rezago_horas": 24},
        )
        self.assertEqual(
            train.resolver_hiperparametros("p3", "baseline", {"n_clusters": "none"}),
            {"n_clusters": None},
        )
        self.assertEqual(train.convertir_valor("0.05"), 0.05)
        self.assertEqual(train.convertir_valor("rf"), "rf")

    def test_parametros_invalidos_se_rechazan(self):
        with self.assertRaises(ValueError):
            train.resolver_hiperparametros("p1", "baseline", {"n_estimators": "300"})
        with self.assertRaises(ValueError):
            train.resolver_hiperparametros("p1", "modelo_inexistente", {})
        with self.assertRaises(ValueError):
            train.planear_tareas("p9", "config", {})

    def test_all_reproduce_los_cuatro_baselines(self):
        sin_cambios = {"rezago_horas": "config", "n_clusters": "config"}
        tareas = train.planear_tareas("all", "config", sin_cambios)
        self.assertEqual(
            [(t.pregunta, t.hiperparametros.get("rezago_horas")) for t in tareas],
            [("p1", 168), ("p1", 24), ("p2", None), ("p3", None)],
        )
        self.assertEqual(train.resolver_etapa("config", tareas), "baseline")
        with self.assertRaises(ValueError):
            train.planear_tareas("all", "config", {"rezago_horas": "24"})

    def test_etapa_de_trabajo(self):
        tareas = train.planear_tareas("p2", "config", {})
        self.assertEqual(train.resolver_etapa("search", tareas), "search")
        with self.assertRaises(ValueError):
            train.resolver_etapa("produccion", tareas)

    def test_cada_modelo_configurado_tiene_entrenador_y_familia(self):
        from mibici.config import FAMILIA_MODELO

        for pregunta, modelos in HIPERPARAMETROS.items():
            for modelo in modelos:
                self.assertIn((pregunta, modelo), train.ENTRENADORES)
                self.assertIn((pregunta, modelo), FAMILIA_MODELO)


if __name__ == "__main__":
    unittest.main()
