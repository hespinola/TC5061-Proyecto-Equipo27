"""Feature engineering (``mibici/features.py``): casos normales y casos límite.

Funciones bajo prueba: ``agregar_variables_calendario``, ``agregar_rezagos``,
``agregar_promedios_moviles``, ``construir_objetivo_riesgo`` y ``filtrar_periodo``.

Trabajan con paneles sintéticos de unas pocas filas, de modo que cada valor esperado
puede verificarse a mano. Ejecución (desde ``tec-mlops-equipo-27/``)::

    python -m pytest tests/test_feature_engineering.py -v
"""

import numpy as np
import pandas as pd
import pytest

from mibici import features as f

LUNES = "2026-03-02"  # lunes


def panel_horario(valores=None, estacion=1, inicio=f"{LUNES} 00:00", **columnas) -> pd.DataFrame:
    """Panel horario completo de una estación; ``valores`` es la serie de ``salidas``."""
    valores = np.arange(10) if valores is None else np.asarray(valores)
    panel = pd.DataFrame(
        {
            f.COL_ESTACION: estacion,
            f.COL_PERIODO: pd.date_range(inicio, periods=len(valores), freq="h"),
            "salidas": valores,
        }
    )
    for nombre, valor in columnas.items():
        panel[nombre] = valor
    return panel


# ---------------------------------------------------------------------------
# agregar_variables_calendario
# ---------------------------------------------------------------------------
def test_calendario_caso_normal_lunes_a_medianoche():
    panel = pd.DataFrame({f.COL_PERIODO: pd.to_datetime([f"{LUNES} 00:00"])})
    fila = f.agregar_variables_calendario(panel).iloc[0]

    assert (fila.hora_del_dia, fila.dia_semana, fila.es_fin_de_semana, fila.mes) == (0, 0, 0, 3)
    assert fila.hora_sin == pytest.approx(0.0, abs=1e-12)
    assert fila.hora_cos == pytest.approx(1.0)
    assert fila.dia_sin == pytest.approx(0.0, abs=1e-12)
    assert fila.dia_cos == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fecha", "es_fin_de_semana"),
    [
        ("2026-03-02", 0),  # lunes
        ("2026-03-06", 0),  # viernes: último día laboral
        ("2026-03-07", 1),  # sábado
        ("2026-03-08", 1),  # domingo
    ],
)
def test_calendario_fin_de_semana(fecha, es_fin_de_semana):
    panel = pd.DataFrame({f.COL_PERIODO: pd.to_datetime([f"{fecha} 12:00"])})
    assert f.agregar_variables_calendario(panel).es_fin_de_semana.iloc[0] == es_fin_de_semana


def test_calendario_codificacion_ciclica_hace_contiguas_las_23_y_las_0():
    horas = pd.to_datetime([f"{LUNES} 23:00", f"{LUNES} 00:00", f"{LUNES} 01:00"])
    out = f.agregar_variables_calendario(pd.DataFrame({f.COL_PERIODO: horas}))
    puntos = out[["hora_sin", "hora_cos"]].to_numpy()

    distancia_23_a_0 = np.linalg.norm(puntos[0] - puntos[1])
    distancia_0_a_1 = np.linalg.norm(puntos[1] - puntos[2])
    assert distancia_23_a_0 == pytest.approx(distancia_0_a_1)


def test_calendario_festivos():
    horas = pd.date_range(f"{LUNES} 00:00", periods=48, freq="h")
    panel = pd.DataFrame({f.COL_PERIODO: horas})

    con_festivo = f.agregar_variables_calendario(panel, festivos=[f"{LUNES} 15:30"])
    assert con_festivo.es_festivo.iloc[:24].eq(1).all()  # el día completo, aunque traiga hora
    assert con_festivo.es_festivo.iloc[24:].eq(0).all()

    assert f.agregar_variables_calendario(panel, festivos=[]).es_festivo.eq(0).all()  # límite
    assert "es_festivo" not in f.agregar_variables_calendario(panel).columns


def test_calendario_no_modifica_el_panel_recibido():
    panel = pd.DataFrame({f.COL_PERIODO: pd.to_datetime([f"{LUNES} 08:00"])})
    f.agregar_variables_calendario(panel)
    assert list(panel.columns) == [f.COL_PERIODO]


# ---------------------------------------------------------------------------
# agregar_rezagos
# ---------------------------------------------------------------------------
def test_rezagos_caso_normal_desplaza_k_periodos():
    out = f.agregar_rezagos(panel_horario(np.arange(10)), columnas=["salidas"], rezagos=[3])

    esperado = [np.nan] * 3 + [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np.testing.assert_array_equal(out["salidas_rezago_3"].to_numpy(), esperado)


def test_rezagos_no_mezclan_estaciones():
    panel = pd.concat(
        [panel_horario(np.arange(5), estacion=1), panel_horario(np.arange(100, 105), estacion=2)]
    )
    out = f.agregar_rezagos(panel, columnas=["salidas"], rezagos=[1])

    primera_hora = out.groupby(f.COL_ESTACION).head(1)["salidas_rezago_1"]
    assert primera_hora.isna().all()  # la primera hora de la estación 2 no hereda de la 1
    assert out.loc[out[f.COL_ESTACION] == 2, "salidas_rezago_1"].tolist()[1:] == [100, 101, 102, 103]


def test_rezagos_no_dependen_del_orden_de_las_filas():
    ordenado = f.agregar_rezagos(panel_horario(np.arange(8)), columnas=["salidas"], rezagos=[2])
    revuelto = panel_horario(np.arange(8)).sample(frac=1, random_state=0)
    out = f.agregar_rezagos(revuelto, columnas=["salidas"], rezagos=[2])
    pd.testing.assert_frame_equal(out, ordenado)


def test_rezago_menor_al_horizonte_se_rechaza_por_fuga_de_informacion():
    with pytest.raises(ValueError, match="menores al horizonte"):
        f.agregar_rezagos(panel_horario(), columnas=["salidas"], rezagos=[1, 24], horizonte=3)


def test_rezago_igual_al_horizonte_es_valido():  # límite
    out = f.agregar_rezagos(panel_horario(), columnas=["salidas"], rezagos=[3], horizonte=3)
    assert "salidas_rezago_3" in out.columns


@pytest.mark.parametrize(
    ("rezagos", "horizonte", "mensaje"),
    [
        ([], 1, "al menos un rezago"),
        ([0], 1, "entero positivo"),
        ([-2], 1, "entero positivo"),
        ([1.5], 1, "entero positivo"),
        ([True], 1, "entero positivo"),
        ([1], 0, "entero positivo"),
    ],
)
def test_rezagos_parametros_invalidos(rezagos, horizonte, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        f.agregar_rezagos(panel_horario(), ["salidas"], rezagos, horizonte)


def test_rezagos_rechazan_panel_con_huecos():
    panel = panel_horario(np.arange(6)).drop(index=3)
    with pytest.raises(ValueError, match="sin huecos"):
        f.agregar_rezagos(panel, columnas=["salidas"], rezagos=[1])


def test_rezagos_rechazan_claves_duplicadas():
    panel = pd.concat([panel_horario(np.arange(4)), panel_horario(np.arange(4)).iloc[[0]]])
    with pytest.raises(ValueError, match="una sola fila por estación y periodo"):
        f.agregar_rezagos(panel, columnas=["salidas"], rezagos=[1])


def test_rezagos_exigen_periodo_datetime_y_columnas_clave():
    texto = panel_horario().assign(**{f.COL_PERIODO: lambda t: t[f.COL_PERIODO].astype(str)})
    with pytest.raises(TypeError, match="datetime"):
        f.agregar_rezagos(texto, columnas=["salidas"], rezagos=[1])
    with pytest.raises(KeyError, match="estacion_id"):
        f.agregar_rezagos(panel_horario().drop(columns=f.COL_ESTACION), ["salidas"], [1])


# ---------------------------------------------------------------------------
# agregar_promedios_moviles
# ---------------------------------------------------------------------------
def test_promedios_moviles_caso_normal():
    out = f.agregar_promedios_moviles(
        panel_horario(np.arange(8)), columnas=["salidas"], ventanas=[3], horizonte=1
    )

    # La serie se desplaza 1 hora: en t=3 la ventana es [0, 1, 2] y su media es 1.
    esperado = [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(out["salidas_media_3"].to_numpy(), esperado)


def test_promedios_moviles_nunca_incluyen_el_periodo_actual():
    original = panel_horario(np.arange(8))
    modificado = original.copy()
    modificado.loc[modificado.index[-1], "salidas"] = 10_000  # cambia solo la última hora

    a = f.agregar_promedios_moviles(original, ["salidas"], [3])
    b = f.agregar_promedios_moviles(modificado, ["salidas"], [3])

    pd.testing.assert_series_equal(a["salidas_media_3"], b["salidas_media_3"])


def test_promedios_moviles_ventana_mayor_que_la_serie_son_todos_na():  # límite
    out = f.agregar_promedios_moviles(panel_horario(np.arange(5)), ["salidas"], [24])
    assert out["salidas_media_24"].isna().all()


def test_promedios_moviles_no_cruzan_estaciones():
    panel = pd.concat(
        [panel_horario(np.arange(6), estacion=1), panel_horario(np.arange(6), estacion=2)]
    )
    out = f.agregar_promedios_moviles(panel, ["salidas"], [3])
    por_estacion = out.groupby(f.COL_ESTACION)["salidas_media_3"].apply(lambda s: s.isna().sum())
    assert por_estacion.tolist() == [3, 3]  # cada estación arranca con su propio calentamiento


@pytest.mark.parametrize("ventanas", [[], [0], [-1], [2.5]])
def test_promedios_moviles_ventanas_invalidas(ventanas):
    with pytest.raises(ValueError):
        f.agregar_promedios_moviles(panel_horario(), ["salidas"], ventanas)


# ---------------------------------------------------------------------------
# construir_objetivo_riesgo
# ---------------------------------------------------------------------------
@pytest.fixture
def panel_flujo() -> pd.DataFrame:
    """Flujo neto con ventanas de 3 h cuyos acumulados futuros se calculan a mano."""
    flujo = [-2, -2, -1, 1, 2, 2, 0, 0, 0]
    return panel_horario(flujo, flujo_neto=flujo, capacidad=10)


def test_objetivo_riesgo_caso_normal(panel_flujo):
    out = f.construir_objetivo_riesgo(panel_flujo, horizonte=3, umbral=0.5)  # límite = 5 bicis

    # Flujo futuro de 3 h por fila: -5, -2, 2, 5, 4, 2, 0 y las dos últimas incompletas.
    assert out["objetivo_flujo_futuro"].tolist()[:7] == [-5, -2, 2, 5, 4, 2, 0]
    assert out["objetivo_flujo_futuro"].iloc[7:].isna().all()
    assert out["objetivo_riesgo_sin_bicis"].tolist()[:7] == [1, 0, 0, 0, 0, 0, 0]
    assert out["objetivo_riesgo_sin_anclajes"].tolist()[:7] == [0, 0, 0, 1, 0, 0, 0]
    assert out["objetivo_en_riesgo"].tolist()[:7] == [1, 0, 0, 1, 0, 0, 0]


def test_objetivo_riesgo_cola_incompleta_queda_en_na(panel_flujo):
    out = f.construir_objetivo_riesgo(panel_flujo, horizonte=3, umbral=0.5)

    cola = out.iloc[-2:]
    assert cola["objetivo_en_riesgo"].isna().all()
    assert cola["objetivo_fin_periodo"].isna().all()
    esperado = out[f.COL_PERIODO].iloc[:7] + pd.Timedelta(hours=3)
    pd.testing.assert_series_equal(
        out["objetivo_fin_periodo"].iloc[:7], esperado, check_names=False
    )


def test_objetivo_riesgo_el_umbral_es_inclusivo():
    flujo = [5, 0, 0, 4, 0, 0]  # 5 = 0.5 x 10 exacto; 4 no alcanza
    out = f.construir_objetivo_riesgo(
        panel_horario(flujo, flujo_neto=flujo, capacidad=10), horizonte=3, umbral=0.5
    )
    assert out["objetivo_riesgo_sin_anclajes"].tolist()[:4] == [1, 0, 0, 0]


@pytest.mark.parametrize("capacidad", [0, -5, np.nan, np.inf])
def test_objetivo_riesgo_sin_capacidad_valida_no_se_etiqueta(panel_flujo, capacidad):
    panel_flujo["capacidad"] = capacidad
    out = f.construir_objetivo_riesgo(panel_flujo, horizonte=3, umbral=0.5)
    assert out["objetivo_en_riesgo"].isna().all()


def test_objetivo_riesgo_no_cruza_estaciones():
    flujo = [-4, -4, -4]
    panel = pd.concat(
        [
            panel_horario(flujo, estacion=1, flujo_neto=flujo, capacidad=10),
            panel_horario([9, 9, 9], estacion=2, flujo_neto=[9, 9, 9], capacidad=10),
        ]
    )
    out = f.construir_objetivo_riesgo(panel, horizonte=2, umbral=0.5).set_index(
        [f.COL_ESTACION, f.COL_PERIODO]
    )
    # Estación 1: -4-4 = -8 (sin bicis). Estación 2: 18 (sin anclajes); sin mezcla entre ambas.
    assert out.loc[1, "objetivo_flujo_futuro"].tolist()[:2] == [-8, -8]
    assert out.loc[2, "objetivo_flujo_futuro"].tolist()[:2] == [18, 18]


@pytest.mark.parametrize("umbral", [0, -0.1, 1.01, np.nan, np.inf])
def test_objetivo_riesgo_umbral_invalido(panel_flujo, umbral):
    with pytest.raises(ValueError, match="umbral debe estar entre 0"):
        f.construir_objetivo_riesgo(panel_flujo, horizonte=3, umbral=umbral)


def test_objetivo_riesgo_umbral_uno_es_valido(panel_flujo):  # límite
    out = f.construir_objetivo_riesgo(panel_flujo, horizonte=3, umbral=1)
    assert out["objetivo_en_riesgo"].dropna().eq(0).all()  # nadie llega a 10 bicis en 3 h


@pytest.mark.parametrize("horizonte", [0, -1, 1.5, True])
def test_objetivo_riesgo_horizonte_invalido(panel_flujo, horizonte):
    with pytest.raises(ValueError, match="entero positivo"):
        f.construir_objetivo_riesgo(panel_flujo, horizonte=horizonte)


def test_objetivo_riesgo_exige_flujo_y_capacidad(panel_flujo):
    with pytest.raises(KeyError, match="capacidad"):
        f.construir_objetivo_riesgo(panel_flujo.drop(columns="capacidad"))


# ---------------------------------------------------------------------------
# filtrar_periodo
# ---------------------------------------------------------------------------
@pytest.fixture
def tres_dias() -> pd.DataFrame:
    return panel_horario(np.arange(72), inicio=f"{LUNES} 00:00")


def test_filtrar_periodo_extremos_son_fechas_inclusivas(tres_dias):
    out = f.filtrar_periodo(tres_dias, inicio="2026-03-03", fin="2026-03-03")

    assert len(out) == 24  # el día 3 completo
    assert out[f.COL_PERIODO].min() == pd.Timestamp("2026-03-03 00:00")
    assert out[f.COL_PERIODO].max() == pd.Timestamp("2026-03-03 23:00")


def test_filtrar_periodo_sin_limites_conserva_todo_y_devuelve_copia(tres_dias):
    out = f.filtrar_periodo(tres_dias)
    assert len(out) == len(tres_dias)
    out["salidas"] = -1
    assert (tres_dias["salidas"] >= 0).all()  # la copia no altera el original


@pytest.mark.parametrize(
    ("inicio", "fin", "esperadas"),
    [
        ("2026-03-02", None, 72),
        (None, "2026-03-02", 24),
        ("2026-03-05", None, 0),  # límite: sin filas
        ("2026-03-03", "2026-03-02", 0),  # límite: rango invertido
    ],
)
def test_filtrar_periodo_limites_abiertos_y_vacios(tres_dias, inicio, fin, esperadas):
    assert len(f.filtrar_periodo(tres_dias, inicio, fin)) == esperadas


def test_filtrar_periodo_purga_ventanas_objetivo_que_cruzan_el_corte():
    flujo = [1] * 48
    panel = panel_horario(flujo, flujo_neto=flujo, capacidad=10)
    con_objetivo = f.construir_objetivo_riesgo(panel, horizonte=3, umbral=0.5)

    out = f.filtrar_periodo(
        con_objetivo, "2026-03-02", "2026-03-02", col_fin_objetivo="objetivo_fin_periodo"
    )

    # Las ventanas de 3 h que empiezan a las 22:00 y 23:00 cierran después de la medianoche y
    # se purgan. La de las 21:00 cierra exactamente en el corte (exclusivo) y se conserva.
    assert out[f.COL_PERIODO].max() == pd.Timestamp("2026-03-02 21:00")
    assert len(out) == 22


def test_filtrar_periodo_descarta_ventanas_incompletas_con_col_fin_objetivo():
    flujo = [1] * 5
    panel = panel_horario(flujo, flujo_neto=flujo, capacidad=10)
    con_objetivo = f.construir_objetivo_riesgo(panel, horizonte=3, umbral=0.5)

    out = f.filtrar_periodo(con_objetivo, fin="2026-03-31", col_fin_objetivo="objetivo_fin_periodo")

    assert len(out) == 3  # las dos últimas horas tienen objetivo_fin_periodo en NaT


def test_filtrar_periodo_columna_fin_objetivo_requiere_fin(tres_dias):
    tres_dias["objetivo_fin_periodo"] = tres_dias[f.COL_PERIODO]
    with pytest.raises(ValueError, match="Se requiere fin"):
        f.filtrar_periodo(tres_dias, inicio="2026-03-02", col_fin_objetivo="objetivo_fin_periodo")


def test_filtrar_periodo_columna_inexistente(tres_dias):
    with pytest.raises(KeyError, match="fecha_inexistente"):
        f.filtrar_periodo(tres_dias, "2026-03-02", col="fecha_inexistente")