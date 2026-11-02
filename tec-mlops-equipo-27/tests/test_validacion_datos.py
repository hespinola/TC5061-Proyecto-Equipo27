"""Validación de datos con DeepChecks (mibici/validacion.py).

Cada prueba introduce un problema concreto en un panel sintético limpio y comprueba
que el chequeo correspondiente lo detecta; el panel limpio pasa todos los chequeos.
No requiere datos descargados ni servidor de MLflow:

    python -m pytest tests/test_validacion_datos.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici import features as f
from mibici import validacion as v
from mibici.config import FIN_ENTRENAMIENTO, FIN_VALIDACION

ESTACIONES = 4


def construir_panel(inicio="2025-10-01", fin="2026-10-01", semilla=0):
    """Panel con las mismas columnas que produce ``mibici.preparar``."""
    rng = np.random.default_rng(semilla)
    horas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    panel = pd.DataFrame(
        {
            f.COL_ESTACION: np.repeat(np.arange(1, ESTACIONES + 1), len(horas)),
            f.COL_PERIODO: np.tile(horas, ESTACIONES),
        }
    )
    n = len(panel)
    panel["salidas"] = rng.poisson(1.5, n).astype("int32")
    panel["llegadas"] = rng.poisson(1.5, n).astype("int32")
    panel["flujo_neto"] = panel["llegadas"] - panel["salidas"]
    nomenclatura = pd.DataFrame(
        {
            "id": range(1, ESTACIONES + 1),
            "dpcapacity": [12, 15, 20, 15],
            "latitude": 20.67,
            "longitude": -103.35,
        }
    )
    panel = f.agregar_variables_calendario(panel)
    panel = f.agregar_rezagos(panel)
    panel = f.agregar_promedios_moviles(panel)
    panel = f.agregar_capacidad(panel, nomenclatura)
    panel = f.agregar_flujo_acumulado_dia(panel)
    return f.construir_objetivo_riesgo(panel)


@pytest.fixture(scope="module")
def dc():
    return v.preparar_deepchecks()


@pytest.fixture(scope="module")
def panel():
    return construir_panel()


@pytest.fixture(scope="module")
def entrenamiento(panel):
    return v.particionar(panel).entrenamiento


def estados(resultado) -> dict[str, str]:
    return {fila["chequeo"]: fila["estado"] for fila in v._resumir(resultado)}


def integridad(dc, df):
    return estados(v.suite_integridad(dc).run(v._dataset(dc, df, v.ETIQUETA_P1, "e")))


def comparar(dc, train, test, etiqueta=v.ETIQUETA_P1):
    suite = (
        v.suite_entrenamiento_prueba(dc) if etiqueta == v.ETIQUETA_P1 else v.suite_objetivo_p2(dc)
    )
    return estados(
        suite.run(v._dataset(dc, train, etiqueta, "e"), v._dataset(dc, test, etiqueta, "p"))
    )


def desplazar(df, dias):
    """Misma distribución, fechas posteriores: una 'prueba' sin drift real."""
    out = df.copy()
    for col in (f.COL_PERIODO, "objetivo_fin_periodo"):
        out[col] = out[col] + pd.Timedelta(days=dias)
    return out


# ---------------------------------------------------------------------------
# Compatibilidad y partición
# ---------------------------------------------------------------------------
def test_preparar_deepchecks_es_idempotente():
    primero = v.preparar_deepchecks()
    segundo = v.preparar_deepchecks()
    assert primero is segundo
    assert np.Inf == np.inf


def test_particion_respeta_las_fechas_de_config(panel):
    partes = v.particionar(panel)
    corte = pd.Timestamp(FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)
    inicio_prueba = pd.Timestamp(FIN_VALIDACION) + pd.Timedelta(days=1)
    assert partes.entrenamiento[f.COL_PERIODO].max() < corte
    # Las etiquetas de P2 tampoco cruzan el corte de entrenamiento.
    assert partes.entrenamiento["objetivo_fin_periodo"].max() <= corte
    assert partes.prueba[f.COL_PERIODO].min() == inicio_prueba
    assert partes.prueba["objetivo_en_riesgo"].notna().all()


def test_particion_vacia_levanta_error(panel):
    solo_entrenamiento = panel[panel[f.COL_PERIODO] < pd.Timestamp(FIN_ENTRENAMIENTO)]
    with pytest.raises(ValueError, match="Particiones vacías"):
        v.particionar(solo_entrenamiento)


def test_dataset_exige_las_columnas_del_panel(dc, entrenamiento):
    with pytest.raises(KeyError, match="capacidad"):
        v._dataset(dc, entrenamiento.drop(columns="capacidad"), v.ETIQUETA_P1, "e")


def test_dataset_p2_descarta_filas_sin_etiqueta(dc, entrenamiento):
    df = entrenamiento.copy()
    df.loc[df.index[:10], "objetivo_en_riesgo"] = pd.NA
    ds = v._dataset(dc, df, v.ETIQUETA_P2, "e")
    assert ds.n_samples == len(df) - 10
    assert set(ds.label_col.unique()) <= {0, 1}


# ---------------------------------------------------------------------------
# Integridad: caso normal y problemas que deben detectarse
# ---------------------------------------------------------------------------
def test_panel_limpio_pasa_la_integridad(dc, entrenamiento):
    resultado = integridad(dc, entrenamiento)
    assert set(resultado.values()) == {"ok"}, resultado


def test_detecta_valores_faltantes(dc, entrenamiento):
    df = entrenamiento.copy()
    df.loc[df.sample(frac=0.2, random_state=0).index, "capacidad"] = np.nan
    assert integridad(dc, df)["PercentOfNulls"] == "falla"


def test_nulos_dentro_del_umbral_se_aceptan(dc, entrenamiento):
    df = entrenamiento.copy()
    df.loc[df.sample(frac=0.01, random_state=0).index, "capacidad"] = np.nan
    assert integridad(dc, df)["PercentOfNulls"] == "ok"


def test_detecta_nulos_mezclados(dc, entrenamiento):
    df = entrenamiento.copy()
    df["flujo_acumulado_dia_rel"] = df["flujo_acumulado_dia_rel"].astype(object)
    df.loc[df.index[:20], "flujo_acumulado_dia_rel"] = "NaN"
    df.loc[df.index[20:40], "flujo_acumulado_dia_rel"] = np.nan
    assert integridad(dc, df)["Mixed Nulls"] == "falla"


def test_detecta_tipos_de_dato_mezclados(dc, entrenamiento):
    df = entrenamiento.copy()
    df["capacidad"] = df["capacidad"].astype(object)
    df.loc[df.sample(frac=0.05, random_state=0).index, "capacidad"] = "veinte"
    assert integridad(dc, df)["Mixed Data Types"] == "falla"


def test_detecta_columna_constante(dc, entrenamiento):
    df = entrenamiento.assign(capacidad=15.0)
    assert integridad(dc, df)["Single Value in Column"] == "falla"


def test_detecta_claves_estacion_hora_duplicadas(dc, entrenamiento):
    df = pd.concat([entrenamiento, entrenamiento.head(50)], ignore_index=True)
    assert integridad(dc, df)["Data Duplicates"] == "falla"


def test_detecta_fuga_de_informacion(dc, entrenamiento):
    # Error típico: usar el conteo de la hora actual como si fuera un rezago.
    df = entrenamiento.assign(salidas_rezago_24=entrenamiento["salidas"])
    assert integridad(dc, df)["Feature Label Correlation"] == "falla"


# ---------------------------------------------------------------------------
# Entrenamiento vs. prueba: caso normal y problemas que deben detectarse
# ---------------------------------------------------------------------------
def test_misma_distribucion_en_fechas_posteriores_pasa(dc, entrenamiento):
    prueba = desplazar(entrenamiento, 400)
    resultado = comparar(dc, entrenamiento, prueba)
    assert set(resultado.values()) == {"ok"}, resultado
    assert comparar(dc, entrenamiento, prueba, v.ETIQUETA_P2) == {"Label Drift": "ok"}


def test_detecta_drift_en_la_demanda(dc, entrenamiento):
    prueba = desplazar(entrenamiento, 400)
    for col in v.FEATURES_DRIFT:
        if col.startswith("salidas_"):
            prueba[col] = prueba[col] * 3
    assert comparar(dc, entrenamiento, prueba)["Feature Drift"] == "falla"


def test_detecta_drift_del_objetivo_p1(dc, entrenamiento):
    prueba = desplazar(entrenamiento, 400)
    prueba["salidas"] = prueba["salidas"] + 5
    assert comparar(dc, entrenamiento, prueba)["Label Drift"] == "falla"


def test_detecta_drift_del_objetivo_p2(dc, entrenamiento):
    prueba = desplazar(entrenamiento, 400)
    prueba["objetivo_en_riesgo"] = 1
    assert comparar(dc, entrenamiento, prueba, v.ETIQUETA_P2)["Label Drift"] == "falla"


def test_detecta_traslape_de_fechas_entre_particiones(dc, entrenamiento):
    # Una partición aleatoria (no temporal) mezcla fechas de entrenamiento en prueba.
    prueba = entrenamiento.sample(frac=0.3, random_state=0)
    resultado = comparar(dc, entrenamiento.drop(prueba.index), prueba)
    assert resultado["Date Train-Test Leakage (overlap)"] == "falla"


def test_detecta_estaciones_nuevas_en_prueba(dc, entrenamiento):
    prueba = desplazar(entrenamiento, 400)
    prueba.loc[prueba.sample(frac=0.1, random_state=0).index, f.COL_ESTACION] = 999
    assert comparar(dc, entrenamiento, prueba)["New Category Train Test"] == "falla"


# ---------------------------------------------------------------------------
# Comando de punta a punta
# ---------------------------------------------------------------------------
def test_comando_guarda_resumen_y_reportes(panel, tmp_path):
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    resumen_path = tmp_path / "validacion.json"
    html_dir = tmp_path / "html"
    args = [
        "--panel-path", str(panel_path),
        "--resumen-path", str(resumen_path),
        "--html-dir", str(html_dir),
    ]  # fmt: skip

    resultado = CliRunner().invoke(v.app, args)

    assert resultado.exit_code == 0, resultado.output
    resumen = json.loads(resumen_path.read_text())
    assert set(resumen["suites"]) == {
        "integridad_entrenamiento",
        "integridad_prueba",
        "entrenamiento_vs_prueba_p1",
        "entrenamiento_vs_prueba_p2",
    }
    estados_todos = [c["estado"] for filas in resumen["suites"].values() for c in filas]
    assert "error" not in estados_todos
    assert sorted(p.name for p in html_dir.iterdir()) == sorted(
        f"{nombre}.html" for nombre in resumen["suites"]
    )


def test_modo_estricto_falla_si_hay_hallazgos(panel, tmp_path):
    con_nulos = panel.copy()
    con_nulos.loc[con_nulos.sample(frac=0.3, random_state=0).index, "capacidad"] = np.nan
    panel_path = tmp_path / "panel.parquet"
    con_nulos.to_parquet(panel_path, index=False)
    args = [
        "--panel-path", str(panel_path),
        "--resumen-path", str(tmp_path / "r.json"),
        "--html-dir", str(tmp_path / "html"),
    ]  # fmt: skip

    assert CliRunner().invoke(v.app, args).exit_code == 0
    assert CliRunner().invoke(v.app, [*args, "--estricto"]).exit_code == 1
