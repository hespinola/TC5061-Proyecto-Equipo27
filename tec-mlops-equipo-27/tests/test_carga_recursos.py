"""Pruebas del procesamiento de métricas, sin Docker ni servidor HTTP."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "carga", Path(__file__).resolve().parents[1] / "load_tests/ejecutar.py"
)
carga = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(carga)


@pytest.mark.parametrize(
    "value,expected",
    [("1GiB", 1024**3), ("512 MiB", 512 * 1024**2), ("1.5MB", 1500000), ("0B", 0)],
)
def test_unidades_memoria(value, expected):
    assert carga.memoria_bytes(value) == expected


def test_unidad_desconocida_no_se_convierte_a_cero():
    with pytest.raises(ValueError):
        carga.memoria_bytes("1desconocido")


def test_stats_conserva_cpu_multicore_y_limite():
    row = carga.normalizar({"Name": "api", "CPUPerc": "125.5%", "MemUsage": "128MiB / 1GiB"})
    assert row == {
        "contenedor": "api",
        "cpu_pct": 125.5,
        "memoria_bytes": 128 * 1024**2,
        "limite_bytes": 1024**3,
    }


def test_stats_incompletos_fallan_explicitamente():
    with pytest.raises(KeyError):
        carga.normalizar({"Name": "api"})
