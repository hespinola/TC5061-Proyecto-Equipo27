"""Entrenamiento y utilidades (``mibici/modeling``): casos normales y casos límite.

Funciones bajo prueba: ``convertir_valor``, ``resolver_hiperparametros``, ``planear_tareas``,
``resolver_etapa``, ``metricas_clasificacion`` y ``frecuencias_riesgo`` (``train.py``);
``elegir_umbral`` y ``cluster_scores`` (``candidates.py``); y los estimadores
``LagRegressor``, ``FrequencyClassifier`` y ``ThresholdClassifier`` (``estimators.py``).

No usan MLflow ni datos reales. Ejecución (desde ``tec-mlops-equipo-27/``)::

    python -m pytest tests/test_utilidades_entrenamiento.py -v
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mibici.config import HIPERPARAMETROS, UMBRALES_CLASIFICACION
from mibici.modeling import candidates, train
from mibici.modeling.estimators import FrequencyClassifier, LagRegressor, ThresholdClassifier

CONFIG = train.VALOR_DESDE_CONFIG


# ---------------------------------------------------------------------------
# convertir_valor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("none", None),
        ("None", None),
        ("  NULL ", None),
        ("8", 8),
        ("-3", -3),
        (" 42 ", 42),  # espacios alrededor
        ("0.05", 0.05),
        ("1e-3", 0.001),
        ("rf", "rf"),
        ("True", "True"),  # no se interpreta como booleano
        ("", ""),  # límite: cadena vacía
        (8, 8),  # límite: no llega como texto
    ],
)
def test_convertir_valor(texto, esperado):
    resultado = train.convertir_valor(texto)
    assert resultado == esperado
    assert type(resultado) is type(esperado)


# ---------------------------------------------------------------------------
# resolver_hiperparametros
# ---------------------------------------------------------------------------
def test_hiperparametros_sin_cambios_devuelven_los_de_config():
    assert train.resolver_hiperparametros("p1", "hgb", {}) == HIPERPARAMETROS["p1"]["hgb"]


def test_hiperparametros_se_convierten_y_none_es_valido():
    hp = train.resolver_hiperparametros(
        "p1", "hgb", {"n_estimators": "160", "max_depth": "none", "learning_rate": "0.08"}
    )
    assert hp == {"n_estimators": 160, "max_depth": None, "learning_rate": 0.08}


def test_hiperparametros_valor_config_conserva_el_valor_por_defecto():
    hp = train.resolver_hiperparametros("p1", "hgb", {"n_estimators": CONFIG, "max_depth": "3"})
    assert hp["n_estimators"] == HIPERPARAMETROS["p1"]["hgb"]["n_estimators"]
    assert hp["max_depth"] == 3


def test_hiperparametros_no_modifican_la_configuracion_global():
    copia = {k: dict(v) for k, v in HIPERPARAMETROS["p1"].items()}
    hp = train.resolver_hiperparametros("p1", "hgb", {"n_estimators": "999"})
    hp["max_depth"] = -1
    assert HIPERPARAMETROS["p1"] == copia


@pytest.mark.parametrize(
    ("pregunta", "modelo", "valores", "mensaje"),
    [
        ("p9", "baseline", {}, "no válida"),
        ("p1", "modelo_inexistente", {}, "no disponible"),
        ("p1", "baseline", {"n_estimators": "300"}, "no aplica"),
        ("p2", "baseline", {"cualquiera": "1"}, "ninguno"),  # límite: modelo sin hiperparámetros
        ("", "baseline", {}, "no válida"),
    ],
)
def test_hiperparametros_invalidos(pregunta, modelo, valores, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        train.resolver_hiperparametros(pregunta, modelo, valores)


# ---------------------------------------------------------------------------
# planear_tareas y resolver_etapa
# ---------------------------------------------------------------------------
def test_planear_tareas_normaliza_mayusculas_y_espacios():
    (tarea,) = train.planear_tareas(" P1 ", " HGB ", {"n_estimators": "50"})
    assert (tarea.pregunta, tarea.modelo) == ("p1", "hgb")
    assert tarea.hiperparametros["n_estimators"] == 50


def test_planear_tareas_modelo_config_usa_el_baseline():
    (tarea,) = train.planear_tareas("p3", CONFIG, {})
    assert tarea.modelo == "baseline"
    assert tarea.hiperparametros == {"n_clusters": None}


def test_planear_tareas_all_solo_admite_baselines_sin_cambios():
    with pytest.raises(ValueError, match="solo ejecuta los baselines"):
        train.planear_tareas("all", "hgb", {})
    with pytest.raises(ValueError, match="solo ejecuta los baselines"):
        train.planear_tareas("all", CONFIG, {"rezago_horas": "24"})


def test_tarea_es_inmutable():
    (tarea,) = train.planear_tareas("p2", CONFIG, {})
    with pytest.raises(FrozenInstanceError):
        tarea.modelo = "otro"


@pytest.mark.parametrize(
    ("etapa", "modelos", "esperada"),
    [
        (CONFIG, ["baseline"], "baseline"),
        (CONFIG, ["baseline", "hgb"], "candidate"),
        (CONFIG, [], "baseline"),  # límite: sin tareas
        ("search", ["hgb"], "search"),  # la etapa explícita manda
    ],
)
def test_resolver_etapa(etapa, modelos, esperada):
    tareas = [train.Tarea("p1", m, {}) for m in modelos]
    assert train.resolver_etapa(etapa, tareas) == esperada


def test_resolver_etapa_invalida():
    with pytest.raises(ValueError, match="stage_of_work"):
        train.resolver_etapa("produccion", [])


# ---------------------------------------------------------------------------
# metricas_clasificacion
# ---------------------------------------------------------------------------
def test_metricas_clasificacion_valores_calculados_a_mano():
    # Predicción con umbral 0.5: [1, 0, 1, 0] -> 1 acierto de 2 positivos y 2 predichos.
    m = train.metricas_clasificacion([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)

    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["average_precision"] == pytest.approx(0.5 * 1 + 0.5 * (2 / 3))
    assert m["positive_rate"] == pytest.approx(0.5)


def test_metricas_clasificacion_separacion_perfecta():
    m = train.metricas_clasificacion([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert (m["precision"], m["recall"], m["f1"], m["average_precision"]) == (1, 1, 1, 1)


def test_metricas_clasificacion_el_umbral_es_inclusivo():
    m = train.metricas_clasificacion([1, 0], [0.5, 0.1], umbral=0.5)
    assert m["recall"] == 1.0  # prob == umbral cuenta como positivo


def test_metricas_clasificacion_sin_predicciones_positivas_no_divide_entre_cero():
    m = train.metricas_clasificacion([1, 0, 1], [0.2, 0.1, 0.3], umbral=0.99)
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)


def test_metricas_clasificacion_acepta_booleanos():
    m = train.metricas_clasificacion([True, False, True, False], [0.9, 0.1, 0.8, 0.2], 0.5)
    assert m["f1"] == 1.0


@pytest.mark.parametrize("y", [[0, 0, 0], [1, 1], [], [0, 1, 2]])
def test_metricas_clasificacion_exige_exactamente_dos_clases(y):
    with pytest.raises(ValueError, match="ambas clases"):
        train.metricas_clasificacion(y, [0.5] * len(y), 0.5)


# ---------------------------------------------------------------------------
# frecuencias_riesgo
# ---------------------------------------------------------------------------
def test_frecuencias_riesgo_usa_tasa_historica_y_cae_al_promedio_global():
    entrenamiento = pd.DataFrame(
        {
            "estacion_id": [1, 1, 2],
            "hora_del_dia": [8, 8, 9],
            "objetivo_en_riesgo": [1, 0, 1],
        }
    )
    datos = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3, 1],  # la estación 3 no se vio en entrenamiento
            "hora_del_dia": [8, 9, 8, 9],  # la hora 9 no se vio para la estación 1
        }
    )

    tasas = train.frecuencias_riesgo(entrenamiento, datos)

    np.testing.assert_allclose(tasas, [0.5, 1.0, 2 / 3, 2 / 3])


# ---------------------------------------------------------------------------
# elegir_umbral
# ---------------------------------------------------------------------------
def test_elegir_umbral_en_empate_prefiere_el_mayor():
    # Con umbrales de 0.1 a 0.5 la clasificación es perfecta: gana el más alto.
    umbral, tabla = candidates.elegir_umbral([0, 0, 1, 1], [0.05, 0.06, 0.6, 0.9])

    assert umbral == 0.5
    assert [fila["umbral"] for fila in tabla] == list(UMBRALES_CLASIFICACION)


def test_elegir_umbral_puede_ser_el_mas_bajo():  # límite
    umbral, _ = candidates.elegir_umbral([1, 1, 0], [0.012, 0.011, 0.001])
    assert umbral == min(UMBRALES_CLASIFICACION)


def test_elegir_umbral_con_una_sola_clase_levanta_error():
    with pytest.raises(ValueError, match="ambas clases"):
        candidates.elegir_umbral([0, 0, 0], [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# cluster_scores
# ---------------------------------------------------------------------------
COLAPSADO = {"silhouette": -1.0, "davies_bouldin": 1e12, "collapsed": 1}


def test_cluster_scores_grupos_bien_separados():
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)

    puntajes = candidates.cluster_scores(X, np.array([0, 0, 1, 1]))

    assert puntajes["collapsed"] == 0
    assert puntajes["silhouette"] > 0.9
    assert 0 < puntajes["davies_bouldin"] < 0.2


@pytest.mark.parametrize(
    "etiquetas",
    [
        [0, 0, 0, 0],  # un solo grupo
        [0, 1, 2, 3],  # cada punto es su propio grupo
    ],
)
def test_cluster_scores_particion_colapsada_se_marca_en_lugar_de_ocultarse(etiquetas):
    X = np.arange(8, dtype=float).reshape(4, 2)
    assert candidates.cluster_scores(X, np.array(etiquetas)) == COLAPSADO


def test_cluster_scores_una_sola_muestra_esta_colapsada():  # límite
    assert candidates.cluster_scores(np.zeros((1, 2)), np.array([0])) == COLAPSADO


# ---------------------------------------------------------------------------
# Estimadores
# ---------------------------------------------------------------------------
@pytest.fixture
def X_frecuencias() -> pd.DataFrame:
    return pd.DataFrame({"estacion_id": [1, 1, 1, 2], "hora_del_dia": [8, 8, 8, 9]})


def test_lag_regressor_devuelve_la_columna_de_rezago_como_float():
    X = pd.DataFrame({"salidas_rezago_168": [3, 5], "salidas_rezago_24": [1, 2]})

    np.testing.assert_array_equal(LagRegressor().fit(X).predict(X), [3.0, 5.0])
    np.testing.assert_array_equal(LagRegressor("salidas_rezago_24").fit(X).predict(X), [1.0, 2.0])
    assert LagRegressor().fit(X).predict(X).dtype == float


def test_lag_regressor_conserva_los_na_y_exige_la_columna():
    X = pd.DataFrame({"salidas_rezago_168": [np.nan, 4.0]})
    assert np.isnan(LagRegressor().fit(X).predict(X)[0])
    with pytest.raises(KeyError):
        LagRegressor("columna_inexistente").fit(X).predict(X)


def test_frequency_classifier_probabilidades_y_fallback(X_frecuencias):
    y = [1, 0, 0, 1]  # (1, 8): 1/3 de positivos; (2, 9): 1 de 1
    modelo = FrequencyClassifier().fit(X_frecuencias, y)
    nuevo = pd.DataFrame({"estacion_id": [1, 2, 9], "hora_del_dia": [8, 9, 3]})  # el 9 no existe

    proba = modelo.predict_proba(nuevo)

    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 1], [1 / 3, 1.0, 0.5])  # fallback = media global (0.5)
    assert modelo.classes_.tolist() == [0, 1]


def test_frequency_classifier_predict_usa_umbral_inclusivo_de_medio():
    X = pd.DataFrame({"estacion_id": [1, 1], "hora_del_dia": [8, 8]})
    modelo = FrequencyClassifier().fit(X, [1, 0])  # tasa exactamente 0.5
    assert modelo.predict(X).tolist() == [1, 1]


@pytest.mark.parametrize(
    ("umbral", "esperado"),
    [
        (0.0, [1, 1, 1, 1]),  # límite inferior: todo es positivo
        (1 / 3, [1, 1, 1, 1]),  # igualdad exacta cuenta como positivo
        (0.34, [0, 0, 0, 1]),
        (1.0, [0, 0, 0, 1]),  # solo la tasa 1.0 alcanza el umbral 1
        (1.01, [0, 0, 0, 0]),  # límite superior: nada alcanza
    ],
)
def test_threshold_classifier_aplica_el_umbral(X_frecuencias, umbral, esperado):
    base = FrequencyClassifier().fit(X_frecuencias, [1, 0, 0, 1])
    modelo = ThresholdClassifier(base, umbral)

    assert modelo.predict(X_frecuencias).tolist() == esperado
    np.testing.assert_allclose(  # predict_proba no cambia con el umbral
        modelo.predict_proba(X_frecuencias), base.predict_proba(X_frecuencias)
    )