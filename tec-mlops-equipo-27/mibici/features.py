"""Feature engineering reutilizable para el proyecto MiBici.

Las funciones parten de la tabla de viajes limpia que produce ``dataset.py`` y
construyen un panel ``estación x periodo`` (por defecto, una fila por estación y
hora). Sobre ese panel se agregan las variables de cada pregunta:

* **P1 (regresión):** ¿cuántos viajes se iniciarán en una estación en un periodo
  futuro? Objetivo: ``salidas`` del periodo. Variables: calendario, rezagos y
  promedios móviles de la demanda.
* **P2 (clasificación):** ¿una estación estará en riesgo de quedarse sin bicicletas
  o sin anclajes? MiBici no publica el inventario, así que el riesgo se aproxima
  con el flujo neto (llegadas - salidas) relativo a la capacidad de la estación.
* **P3 (clustering):** ¿qué grupos de estaciones comparten patrones de uso? Se
  describe cada estación con su perfil horario de uso y su ubicación.

Regla contra fuga de información: toda variable de un periodo ``t`` usa solo datos
anteriores a ``t``. Los rezagos y ventanas se desplazan al menos ``horizonte``
periodos. Las únicas columnas que miran al futuro son los *objetivos* (prefijo
``objetivo_``), que nunca deben usarse como variables de entrada.

Columnas esperadas en la tabla de viajes (nombres normalizados por ``dataset.py``):
``inicio_del_viaje``, ``fin_del_viaje``, ``origen_id`` y ``destino_id``.
Columnas esperadas en la nomenclatura: ``id``, ``latitude``, ``longitude`` y
``dpcapacity``.
"""

from collections.abc import Iterable
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()

# Columnas de la tabla de viajes limpia.
COL_INICIO = "inicio_del_viaje"
COL_FIN = "fin_del_viaje"
COL_ORIGEN = "origen_id"
COL_DESTINO = "destino_id"

# Columnas de la nomenclatura de estaciones.
NOM_ID = "id"
NOM_LATITUD = "latitude"
NOM_LONGITUD = "longitude"
NOM_CAPACIDAD = "dpcapacity"

# Columnas del panel estación x periodo.
COL_ESTACION = "estacion_id"
COL_PERIODO = "periodo"


def _validar_columnas(df: pd.DataFrame, columnas: Iterable[str], nombre: str) -> None:
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise KeyError(f"A la tabla {nombre} le faltan las columnas {faltantes}.")


# ---------------------------------------------------------------------------
# Panel base (P1, P2 y P3)
# ---------------------------------------------------------------------------
def construir_panel_estacion_periodo(
    viajes: pd.DataFrame,
    freq: str = "h",
    estaciones: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Cuenta salidas y llegadas por estación y periodo, con filas en cero incluidas.

    Las salidas se asignan a la estación de origen según la hora de inicio y las
    llegadas a la estación de destino según la hora de fin. El panel queda
    completo: si una estación no tuvo viajes en una hora, aparece con 0 en lugar de
    faltar, lo que es necesario para calcular rezagos correctos.

    Pertinencia: ``salidas`` es la variable objetivo de P1 y ``flujo_neto``
    (llegadas - salidas) es la base para aproximar el riesgo de P2 y los perfiles
    de P3.

    Args:
        viajes: Tabla de viajes limpia.
        freq: Granularidad temporal en notación de pandas (``"h"``, ``"30min"``, ``"D"``).
        estaciones: Estaciones a incluir. Si es ``None``, se usan todas las que
            aparecen como origen o destino.

    Returns:
        DataFrame ordenado por estación y periodo con las columnas
        ``estacion_id``, ``periodo``, ``salidas``, ``llegadas`` y ``flujo_neto``.
    """
    _validar_columnas(viajes, [COL_INICIO, COL_FIN, COL_ORIGEN, COL_DESTINO], "de viajes")

    salidas = (
        viajes.assign(**{COL_PERIODO: viajes[COL_INICIO].dt.floor(freq)})
        .groupby([COL_ORIGEN, COL_PERIODO])
        .size()
        .rename("salidas")
        .rename_axis([COL_ESTACION, COL_PERIODO])
    )
    llegadas = (
        viajes.assign(**{COL_PERIODO: viajes[COL_FIN].dt.floor(freq)})
        .groupby([COL_DESTINO, COL_PERIODO])
        .size()
        .rename("llegadas")
        .rename_axis([COL_ESTACION, COL_PERIODO])
    )

    if estaciones is None:
        estaciones = pd.Index(viajes[COL_ORIGEN]).union(pd.Index(viajes[COL_DESTINO]))
    periodos = pd.date_range(
        viajes[COL_INICIO].min().floor(freq),
        viajes[COL_FIN].max().floor(freq),
        freq=freq,
    )
    malla = pd.MultiIndex.from_product(
        [sorted(set(estaciones)), periodos], names=[COL_ESTACION, COL_PERIODO]
    )

    panel = (
        pd.concat([salidas, llegadas], axis=1)
        .reindex(malla, fill_value=0)
        .fillna(0)
        .astype("int32")
        .reset_index()
    )
    panel["flujo_neto"] = panel["llegadas"] - panel["salidas"]
    return panel.sort_values([COL_ESTACION, COL_PERIODO], ignore_index=True)


def agregar_variables_calendario(
    panel: pd.DataFrame, festivos: Iterable[str] | None = None
) -> pd.DataFrame:
    """Agrega hora del día, día de la semana, fin de semana, mes y festivos.

    La hora y el día se codifican también con seno y coseno para que el modelo
    entienda que las 23 h y las 0 h son contiguas. Son variables conocidas de
    antemano para cualquier periodo, así que no generan fuga.

    Pertinencia: la demanda de MiBici sigue picos de traslado laboral (mañana y
    tarde entre semana) y un patrón distinto en fines de semana y festivos. Sirve a
    P1 y P2 directamente y a P3 para construir los perfiles.

    Args:
        panel: Panel con la columna ``periodo``.
        festivos: Fechas festivas (``"AAAA-MM-DD"``). Si es ``None``, no se agrega
            la columna ``es_festivo``.
    """
    out = panel.copy()
    t = out[COL_PERIODO]
    out["hora_del_dia"] = t.dt.hour.astype("int8")
    out["dia_semana"] = t.dt.dayofweek.astype("int8")  # 0 = lunes
    out["es_fin_de_semana"] = (out["dia_semana"] >= 5).astype("int8")
    out["mes"] = t.dt.month.astype("int8")
    out["hora_sin"] = np.sin(2 * np.pi * out["hora_del_dia"] / 24)
    out["hora_cos"] = np.cos(2 * np.pi * out["hora_del_dia"] / 24)
    out["dia_sin"] = np.sin(2 * np.pi * out["dia_semana"] / 7)
    out["dia_cos"] = np.cos(2 * np.pi * out["dia_semana"] / 7)
    if festivos is not None:
        fechas = pd.to_datetime(list(festivos)).normalize()
        out["es_festivo"] = t.dt.normalize().isin(fechas).astype("int8")
    return out


def agregar_rezagos(
    panel: pd.DataFrame,
    columnas: Iterable[str] = ("salidas", "llegadas"),
    rezagos: Iterable[int] = (24, 168),
    horizonte: int = 1,
) -> pd.DataFrame:
    """Agrega el valor de cada columna ``k`` periodos atrás, por estación.

    Con frecuencia horaria, el rezago 24 es "la misma hora de ayer" y el 168 es "la
    misma hora de la semana pasada", que capturan la estacionalidad diaria y
    semanal de la demanda.

    Para evitar fuga, todo rezago debe ser mayor o igual al ``horizonte`` de
    predicción: si se predice con 3 horas de anticipación, el dato de hace 1 hora
    todavía no existe al momento de predecir.

    Pertinencia: la demanda reciente es el mejor predictor sencillo de la demanda
    futura (P1) y del desbalance de una estación (P2).

    Raises:
        ValueError: Si algún rezago es menor que el horizonte.
    """
    rezagos = list(rezagos)
    if min(rezagos) < horizonte:
        raise ValueError(
            f"Rezagos {rezagos} menores al horizonte {horizonte}: usarían información "
            "que no existe al momento de predecir."
        )
    out = panel.sort_values([COL_ESTACION, COL_PERIODO]).copy()
    grupos = out.groupby(COL_ESTACION, sort=False)
    for col in columnas:
        for k in rezagos:
            out[f"{col}_rezago_{k}"] = grupos[col].shift(k)
    return out


def agregar_promedios_moviles(
    panel: pd.DataFrame,
    columnas: Iterable[str] = ("salidas", "llegadas"),
    ventanas: Iterable[int] = (24, 168),
    horizonte: int = 1,
) -> pd.DataFrame:
    """Agrega el promedio de las últimas ``w`` observaciones conocidas, por estación.

    La serie se desplaza ``horizonte`` periodos antes de promediar, de modo que la
    ventana termina en el último dato disponible al momento de predecir y nunca
    incluye el periodo objetivo.

    Pertinencia: suaviza el ruido de la demanda hora a hora y resume el nivel
    reciente de uso de la estación (P1 y P2).
    """
    out = panel.sort_values([COL_ESTACION, COL_PERIODO]).copy()
    for col in columnas:
        desplazada = out.groupby(COL_ESTACION, sort=False)[col].shift(horizonte)
        for w in ventanas:
            out[f"{col}_media_{w}"] = (
                desplazada.groupby(out[COL_ESTACION], sort=False)
                .rolling(w, min_periods=w)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return out


# ---------------------------------------------------------------------------
# P2: aproximación del riesgo de quedarse sin bicicletas o sin anclajes
# ---------------------------------------------------------------------------
def agregar_capacidad(panel: pd.DataFrame, nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Une la capacidad de anclajes (``dpcapacity``) de cada estación al panel.

    Pertinencia: el mismo desbalance de 10 bicicletas es crítico en una estación de
    12 anclajes y menor en una de 40, por eso P2 mide el flujo relativo a la
    capacidad. Es un atributo fijo de la estación, sin riesgo de fuga.
    """
    _validar_columnas(nomenclatura, [NOM_ID, NOM_CAPACIDAD], "de nomenclatura")
    capacidad = nomenclatura[[NOM_ID, NOM_CAPACIDAD]].rename(
        columns={NOM_ID: COL_ESTACION, NOM_CAPACIDAD: "capacidad"}
    )
    out = panel.merge(capacidad, on=COL_ESTACION, how="left")
    sin_capacidad = out.loc[out["capacidad"].isna(), COL_ESTACION].nunique()
    if sin_capacidad:
        logger.warning(f"{sin_capacidad} estaciones sin capacidad en la nomenclatura.")
    return out


def agregar_flujo_acumulado_dia(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega el flujo neto acumulado del día hasta antes del periodo, sobre capacidad.

    Aproxima qué tan llena o vacía está la estación respecto a como amaneció:
    valores negativos indican que salieron más bicicletas de las que llegaron. El
    acumulado se reinicia cada día bajo el supuesto de que el rebalanceo nocturno
    regresa las estaciones a un nivel similar; el supuesto debe validarse en la
    exploración.

    Es una suma *exclusiva*: el periodo ``t`` solo ve el flujo de los periodos
    anteriores del mismo día, que ya ocurrieron al momento de predecir con
    horizonte 1. Requiere ``capacidad`` (ver :func:`agregar_capacidad`).

    Pertinencia: es la variable más cercana al inventario que MiBici no publica (P2).
    """
    out = panel.sort_values([COL_ESTACION, COL_PERIODO]).copy()
    dia = out[COL_PERIODO].dt.normalize()
    acumulado = out.groupby([out[COL_ESTACION], dia], sort=False)["flujo_neto"].cumsum()
    out["flujo_acumulado_dia"] = acumulado - out["flujo_neto"]
    out["flujo_acumulado_dia_rel"] = out["flujo_acumulado_dia"] / out["capacidad"]
    return out


def construir_objetivo_riesgo(
    panel: pd.DataFrame, horizonte: int = 3, umbral: float = 0.5
) -> pd.DataFrame:
    """Construye la variable objetivo de P2 a partir del flujo neto futuro.

    Para cada estación y periodo ``t`` se suma el flujo neto de ``t`` a
    ``t + horizonte - 1``. Si una estación que estuviera a media capacidad perdiera
    (o recibiera) más de ``umbral x capacidad`` bicicletas en ese lapso, se
    quedaría sin bicicletas (o sin anclajes). Así:

    * ``objetivo_riesgo_sin_bicis = flujo_futuro <= -umbral x capacidad``
    * ``objetivo_riesgo_sin_anclajes = flujo_futuro >= umbral x capacidad``
    * ``objetivo_en_riesgo`` = cualquiera de los dos.

    Estas columnas usan información futura **a propósito**, porque son lo que se
    quiere predecir. No deben entrar como variables al modelo. Las últimas filas de
    cada estación quedan en NA porque su ventana futura está incompleta.

    Pertinencia: el equipo de rebalanceo necesita saber con algunas horas de
    anticipación qué estaciones van a vaciarse o llenarse para mover bicicletas.

    Args:
        panel: Panel con ``flujo_neto`` y ``capacidad``.
        horizonte: Número de periodos hacia adelante que se evalúan.
        umbral: Fracción de la capacidad que define el riesgo.
    """
    _validar_columnas(panel, ["flujo_neto", "capacidad"], "del panel")
    out = panel.sort_values([COL_ESTACION, COL_PERIODO]).copy()
    grupos = out.groupby(COL_ESTACION, sort=False)["flujo_neto"]
    futuro = sum(grupos.shift(-k) for k in range(horizonte))
    limite = umbral * out["capacidad"]
    valido = futuro.notna() & out["capacidad"].notna()

    out["objetivo_flujo_futuro"] = futuro
    out["objetivo_riesgo_sin_bicis"] = (futuro <= -limite).astype("Int8").where(valido)
    out["objetivo_riesgo_sin_anclajes"] = (futuro >= limite).astype("Int8").where(valido)
    out["objetivo_en_riesgo"] = (
        out["objetivo_riesgo_sin_bicis"] | out["objetivo_riesgo_sin_anclajes"]
    ).where(valido)
    return out


# ---------------------------------------------------------------------------
# P3: perfiles de uso por estación para clustering
# ---------------------------------------------------------------------------
def construir_perfil_estaciones(
    panel: pd.DataFrame, nomenclatura: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Resume cada estación en un vector de patrones de uso para P3.

    Variables por estación:

    * ``perfil_lab_hHH`` y ``perfil_fds_hHH``: proporción de las salidas que ocurre
      en cada hora, entre semana y en fin de semana (48 columnas que suman 1 por
      bloque). Distinguen estaciones de traslado laboral, de ocio o mixtas.
    * ``salidas_promedio_dia``: volumen de uso.
    * ``balance_llegadas_salidas``: si la estación es emisora (< 1) o receptora (> 1).
    * ``latitud`` y ``longitud`` si se da la nomenclatura, para ubicar los grupos en
      el mapa y, si el equipo lo decide, incluir la ubicación en el clustering.

    Debe calcularse **solo con el periodo de entrenamiento** para que la
    evaluación posterior no use información futura. Requiere las columnas de
    :func:`agregar_variables_calendario`.
    """
    _validar_columnas(panel, ["hora_del_dia", "es_fin_de_semana"], "del panel")
    tipo_dia = np.where(panel["es_fin_de_semana"] == 1, "fds", "lab")

    salidas_hora = panel.pivot_table(
        index=COL_ESTACION,
        columns=[tipo_dia, panel["hora_del_dia"]],
        values="salidas",
        aggfunc="sum",
        fill_value=0,
    )
    perfiles = []
    for bloque in ("lab", "fds"):
        if bloque not in salidas_hora.columns.get_level_values(0):
            continue
        b = salidas_hora[bloque].reindex(columns=range(24), fill_value=0)
        b = b.div(b.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        b.columns = [f"perfil_{bloque}_h{h:02d}" for h in b.columns]
        perfiles.append(b)

    totales = panel.groupby(COL_ESTACION).agg(
        salidas=("salidas", "sum"), llegadas=("llegadas", "sum")
    )
    dias = panel[COL_PERIODO].dt.normalize().nunique()
    resumen = pd.DataFrame(
        {
            "salidas_promedio_dia": totales["salidas"] / dias,
            "balance_llegadas_salidas": totales["llegadas"]
            / totales["salidas"].replace(0, np.nan),
        }
    )

    perfil = pd.concat([*perfiles, resumen], axis=1)
    if nomenclatura is not None:
        ubicacion = nomenclatura.set_index(NOM_ID)[[NOM_LATITUD, NOM_LONGITUD]].rename(
            columns={NOM_LATITUD: "latitud", NOM_LONGITUD: "longitud"}
        )
        perfil = perfil.join(ubicacion, how="left")
    return perfil.rename_axis(COL_ESTACION).reset_index()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def filtrar_periodo(
    df: pd.DataFrame, inicio: str | None = None, fin: str | None = None, col: str = COL_PERIODO
) -> pd.DataFrame:
    """Filtra filas con ``inicio <= col < fin + 1 día``; los extremos son fechas inclusivas.

    Se usa para separar entrenamiento, validación y prueba en orden temporal con
    las fechas de ``config.py``.
    """
    mascara = pd.Series(True, index=df.index)
    if inicio is not None:
        mascara &= df[col] >= pd.Timestamp(inicio)
    if fin is not None:
        mascara &= df[col] < pd.Timestamp(fin) + pd.Timedelta(days=1)
    return df.loc[mascara]


def construir_features(
    viajes: pd.DataFrame,
    nomenclatura: pd.DataFrame,
    horizonte: int = 1,
    horizonte_riesgo: int = 3,
    umbral_riesgo: float = 0.5,
    festivos: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Encadena el panel y todas las variables de P1 y P2 en un solo paso."""
    panel = construir_panel_estacion_periodo(viajes, estaciones=nomenclatura[NOM_ID])
    panel = agregar_variables_calendario(panel, festivos=festivos)
    panel = agregar_rezagos(panel, horizonte=max(horizonte, 1))
    panel = agregar_promedios_moviles(panel, horizonte=horizonte)
    panel = agregar_capacidad(panel, nomenclatura)
    panel = agregar_flujo_acumulado_dia(panel)
    return construir_objetivo_riesgo(panel, horizonte=horizonte_riesgo, umbral=umbral_riesgo)


@app.command()
def main(
    viajes_path: Path = PROCESSED_DATA_DIR / "viajes.parquet",
    nomenclatura_path: Path = PROCESSED_DATA_DIR / "nomenclatura.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "panel_estacion_hora.parquet",
):
    """Genera el panel estación x hora con las variables de P1 y P2."""
    logger.info(f"Leyendo {viajes_path} y {nomenclatura_path}")
    viajes = pd.read_parquet(viajes_path)
    nomenclatura = pd.read_parquet(nomenclatura_path)

    panel = construir_features(viajes, nomenclatura)
    panel.to_parquet(output_path, index=False)
    logger.success(f"Panel con {len(panel):,} filas guardado en {output_path}")


if __name__ == "__main__":
    app()
