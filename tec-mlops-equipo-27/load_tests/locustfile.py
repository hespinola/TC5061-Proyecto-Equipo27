"""Una solicitud válida equivale a una inferencia de demanda."""

import json
import math
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD = json.loads(
    (Path(__file__).resolve().parents[1] / "examples/predict-demanda.json").read_text()
)


def respuesta_valida(body):
    """Un HTTP 200 no basta: exigir una predicción finita para la fila enviada."""
    try:
        predictions = body["predicciones"]
        value = predictions[0]["salidas_esperadas"]
        return (
            len(predictions) == 1
            and predictions[0]["estacion_id"] == PAYLOAD["instancias"][0]["estacion_id"]
            and isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )
    except (KeyError, IndexError, TypeError):
        return False


class UsuarioDemanda(HttpUser):
    # Modelo cerrado: cada usuario espera entre 0.5 y 1 s después de responder.
    wait_time = between(0.5, 1.0)

    @task
    def predecir(self):
        with self.client.post(
            "/predict/demanda", json=PAYLOAD, timeout=30, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                valid = respuesta_valida(response.json())
            except ValueError:
                valid = False
            if not valid:
                response.failure("Respuesta de predicción inválida")
