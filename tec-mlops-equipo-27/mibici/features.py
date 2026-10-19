"""Feature engineering reutilizable para el proyecto MiBici.

Las funciones parten de la tabla de viajes limpia que produce ``dataset.py`` y
construyen un panel ``estación x periodo`` (por defecto, una fila por estación y
hora). Sobre ese panel se agregan las variables de cada pregunta:

* **P1 (regresión):** al inicio de la hora ``t``, ¿cuántos viajes se iniciarán en
  esa hora? Objetivo: ``salidas`` del periodo. Variables: calendario, rezagos y
  promedios móviles de la demanda.
* **P2 (clasificación):** ¿el flujo neto total de las tres horas que comienzan en
  ``t`` tendrá magnitud de al menos el 50 % de la capacidad? Es un proxy de
  desequilibrio; no observa inventario ni todos los cruces intermedios del umbral.
* **P3 (clustering):** ¿qué grupos de estaciones comparten patrones de uso? Se
  describe cada estación con su perfil horario de uso y su ubicación.

Regla contra fuga de información: los predictores de ``t`` usan solo horas
anteriores a ``t``. Los conteos contemporáneos (``salidas``, ``llegadas`` y
``flujo_neto``) y las columnas ``objetivo_`` no son predictores. La ventana de P2
no implica desplazar todos los predictores tres horas. Al separar particiones,
usar ``filtrar_periodo(..., col_fin_objetivo="objetivo_fin_periodo")`` para P2.

Columnas esperadas en la tabla de viajes (nombres normalizados por ``dataset.py``):
``inicio_del_viaje``, ``fin_del_viaje``, ``origen_id`` y ``destino_id``.
Columnas esperadas en la nomenclatura: ``id``, ``latitude``, ``longitude`` y
``dpcapacity``.
"""

from collections.abc import Iterable
from numbers import Integral
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    PROCESSED_DATA_DIR,
    REZAGOS_HORAS,
    UMBRAL_RIESGO,
    VENTANA_RIESGO_HORAS,
    VENTANAS_MOVILES_HORAS,
)

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


def _validar_entero_positivo(valor: int, nombre: str) -> None:
    if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < 1:
        raise ValueError(f"{nombre} debe ser un entero positivo.")


def _ordenar_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Valida claves únicas y ordena sin depender del índice recibido."""
    _validar_columnas(panel, [COL_ESTACION, COL_PERIODO], "del panel")
    if panel[[COL_ESTACION, COL_PERIODO]].isna().any().any():
        raise ValueError("El panel no puede tener estación o periodo nulos.")
    if not pd.api.types.is_datetime64_any_dtype(panel[COL_PERIODO]):
        raise TypeError("La columna periodo debe ser datetime.")
    if panel.duplicated([COL_ESTACION, COL_PERIODO]).any():
        raise ValueError("El panel debe tener una sola fila por estación y periodo.")
    return panel.sort_values([COL_ESTACION, COL_PERIODO], ignore_index=True)


def _validar_panel_horario(panel: pd.DataFrame) -> None:
    """Los desplazamientos por fila requieren horas consecutivas por estación."""
    diferencias = panel.groupby(COL_ESTACION, sort=False)[COL_PERIODO].diff().dropna()
    if (diferencias != pd.Timedelta(hours=1)).any():
        raise ValueError("Se requiere un panel horario completo, sin huecos por estación.")


def _validar_nomenclatura(nomenclatura: pd.DataFrame, columnas: Iterable[str]) -> None:
    _validar_columnas(nomenclatura, [NOM_ID, *columnas], "de nomenclatura")
    if nomenclatura[NOM_ID].isna().any() or nomenclatura[NOM_ID].duplicated().any():
        raise ValueError("La nomenclatura debe tener identificadores únicos y no nulos.")


def _capacidad_valida(capacidad: pd.Series) -> pd.Series:
    numerica = pd.to_numeric(capacidad, errors="coerce")
    return numerica.where(np.isfinite(numerica) & (numerica > 0))


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
        estaciones: Estaciones adicionales (por ejemplo, de la nomenclatura).
            Siempre se conservan las observadas como origen o destino; nunca se
            eliminan viajes porque una estación no figure en esta lista.

    Returns:
        DataFrame ordenado por estación y periodo con las columnas
        ``estacion_id``, ``periodo``, ``salidas``, ``llegadas`` y ``flujo_neto``.
    """
    _validar_columnas(viajes, [COL_INICIO, COL_FIN, COL_ORIGEN, COL_DESTINO], "de viajes")
    if viajes.empty:
        raise ValueError("La tabla de viajes está vacía.")
    if viajes[[COL_INICIO, COL_FIN, COL_ORIGEN, COL_DESTINO]].isna().any().any():
        raise ValueError("Los viajes deben tener fechas y estaciones no nulas.")
    for col in (COL_INICIO, COL_FIN):
        if not pd.api.types.is_datetime64_any_dtype(viajes[col]):
            raise TypeError(f"La columna {col} debe ser datetime; normalizarla en dataset.py.")
    if (viajes[COL_FIN] < viajes[COL_INICIO]).any():
        raise ValueError("El fin de un viaje no puede ser anterior a su inicio.")

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

    observadas = pd.Index(viajes[COL_ORIGEN].unique()).union(
        pd.Index(viajes[COL_DESTINO].unique())
    )
    if estaciones is None:
        estaciones = observadas
    else:
        adicionales = pd.Index(list(estaciones))
        if adicionales.isna().any():
            raise ValueError("Las estaciones adicionales no pueden tener IDs nulos.")
        faltantes = observadas.difference(adicionales)
        if len(faltantes):
            logger.warning(
                f"{len(faltantes)} estaciones observadas fuera de la lista suministrada; "
                "se conservan sus viajes."
            )
        estaciones = observadas.union(adicionales)
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
    if panel["salidas"].sum() != len(viajes) or panel["llegadas"].sum() != len(viajes):
        raise ValueError("El panel no conserva los totales de viajes.")
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
    rezagos: Iterable[int] = REZAGOS_HORAS,
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
    _validar_entero_positivo(horizonte, "horizonte")
    rezagos = list(rezagos)
    if not rezagos:
        raise ValueError("Debe especificarse al menos un rezago.")
    for k in rezagos:
        _validar_entero_positivo(k, "rezago")
    if min(rezagos) < horizonte:
        raise ValueError(
            f"Rezagos {rezagos} menores al horizonte {horizonte}: usarían información "
            "que no existe al momento de predecir."
        )
    out = _ordenar_panel(panel)
    _validar_panel_horario(out)
    grupos = out.groupby(COL_ESTACION, sort=False)
    for col in columnas:
        for k in rezagos:
            out[f"{col}_rezago_{k}"] = grupos[col].shift(k)
    return out


def agregar_promedios_moviles(
    panel: pd.DataFrame,
    columnas: Iterable[str] = ("salidas", "llegadas"),
    ventanas: Iterable[int] = VENTANAS_MOVILES_HORAS,
    horizonte: int = 1,
) -> pd.DataFrame:
    """Agrega el promedio de las últimas ``w`` observaciones conocidas, por estación.

    La serie se desplaza ``horizonte`` periodos antes de promediar, de modo que la
    ventana termina en el último dato disponible al momento de predecir y nunca
    incluye el periodo objetivo.

    Pertinencia: suaviza el ruido de la demanda hora a hora y resume el nivel
    reciente de uso de la estación (P1 y P2).
    """
    _validar_entero_positivo(horizonte, "horizonte")
    ventanas = list(ventanas)
    if not ventanas:
        raise ValueError("Debe especificarse al menos una ventana.")
    for w in ventanas:
        _validar_entero_positivo(w, "ventana")
    out = _ordenar_panel(panel)
    _validar_panel_horario(out)
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
    capacidad. La nomenclatura es una fotografía de septiembre de 2026, no una
    medición histórica de capacidad. Esa limitación debe documentarse.
    IDs duplicados se rechazan; capacidades no positivas o no finitas quedan en NA.
    """
    _validar_nomenclatura(nomenclatura, [NOM_CAPACIDAD])
    capacidad = nomenclatura[[NOM_ID, NOM_CAPACIDAD]].rename(
        columns={NOM_ID: COL_ESTACION, NOM_CAPACIDAD: "capacidad"}
    )
    capacidad["capacidad"] = _capacidad_valida(capacidad["capacidad"])
    out = panel.merge(capacidad, on=COL_ESTACION, how="left", validate="many_to_one")
    sin_capacidad = out.loc[out["capacidad"].isna(), COL_ESTACION].nunique()
    if sin_capacidad:
        logger.warning(f"{sin_capacidad} estaciones sin capacidad válida en la nomenclatura.")
    return out


def agregar_flujo_acumulado_dia(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega el flujo neto acumulado del día hasta antes del periodo, sobre capacidad.

    Resume el desequilibrio desde medianoche: valores negativos indican más
    salidas que llegadas. El reinicio diario es una convención de agregación;
    no estima inventario ni supone que exista rebalanceo nocturno.

    Es una suma *exclusiva*: el periodo ``t`` solo ve el flujo de los periodos
    anteriores del mismo día, que ya ocurrieron al momento de predecir con
    horizonte 1. Requiere ``capacidad`` (ver :func:`agregar_capacidad`).

    Pertinencia: mide el desequilibrio reciente observado para P2.
    """
    _validar_columnas(panel, ["flujo_neto", "capacidad"], "del panel")
    out = _ordenar_panel(panel)
    _validar_panel_horario(out)
    dia = out[COL_PERIODO].dt.normalize()
    acumulado = out.groupby([out[COL_ESTACION], dia], sort=False)["flujo_neto"].cumsum()
    out["flujo_acumulado_dia"] = acumulado - out["flujo_neto"]
    out["flujo_acumulado_dia_rel"] = out["flujo_acumulado_dia"] / _capacidad_valida(
        out["capacidad"]
    )
    return out


def construir_objetivo_riesgo(
    panel: pd.DataFrame,
    horizonte: int = VENTANA_RIESGO_HORAS,
    umbral: float = UMBRAL_RIESGO,
) -> pd.DataFrame:
    """Construye la variable objetivo de P2 a partir del flujo neto futuro.

    Al inicio de ``t``, se predice el flujo neto total en ``[t, t + horizonte h)``.
    La etiqueta indica un desequilibrio de al menos ``umbral x capacidad``.
    No observa estaciones vacías/llenas ni todos los cruces intermedios del umbral.
    Las dos direcciones del proxy son:

    * ``objetivo_riesgo_sin_bicis = flujo_futuro <= -umbral x capacidad``
    * ``objetivo_riesgo_sin_anclajes = flujo_futuro >= umbral x capacidad``
    * ``objetivo_en_riesgo`` = cualquiera de los dos.

    Estas columnas usan información futura **a propósito**, porque son lo que se
    quiere predecir. No deben entrar como variables al modelo. Las últimas filas de
    cada estación quedan en NA porque su ventana futura está incompleta.
    ``objetivo_fin_periodo`` es el cierre exclusivo de la ventana; permite purgar
    etiquetas que crucen el corte temporal mediante :func:`filtrar_periodo`.

    Pertinencia: ayuda a priorizar estaciones por probabilidad de desequilibrio.

    Args:
        panel: Panel con ``flujo_neto`` y ``capacidad``.
        horizonte: Número de horas de la ventana objetivo (no anticipación de inputs).
        umbral: Fracción de la capacidad que define el riesgo.
    """
    _validar_columnas(panel, ["flujo_neto", "capacidad"], "del panel")
    _validar_entero_positivo(horizonte, "horizonte")
    if not np.isfinite(umbral) or not 0 < umbral <= 1:
        raise ValueError("umbral debe estar entre 0 (exclusivo) y 1 (inclusivo).")
    out = _ordenar_panel(panel)
    _validar_panel_horario(out)
    grupos = out.groupby(COL_ESTACION, sort=False)["flujo_neto"]
    futuro = sum(grupos.shift(-k) for k in range(horizonte))
    limite = umbral * _capacidad_valida(out["capacidad"])
    valido = futuro.notna() & limite.notna()

    out["objetivo_flujo_futuro"] = futuro
    out["objetivo_fin_periodo"] = (out[COL_PERIODO] + pd.Timedelta(hours=horizonte)).where(
        futuro.notna()
    )
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
    panel = _ordenar_panel(panel)
    if panel.empty:
        raise ValueError("El periodo para construir perfiles está vacío.")
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
        # Mantener un esquema de 48 columnas incluso en muestras sin fines de semana.
        if bloque in salidas_hora.columns.get_level_values(0):
            b = salidas_hora[bloque].reindex(columns=range(24), fill_value=0)
        else:
            b = pd.DataFrame(0, index=salidas_hora.index, columns=range(24))
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
        _validar_nomenclatura(nomenclatura, [NOM_LATITUD, NOM_LONGITUD])
        ubicacion = nomenclatura.set_index(NOM_ID)[[NOM_LATITUD, NOM_LONGITUD]].rename(
            columns={NOM_LATITUD: "latitud", NOM_LONGITUD: "longitud"}
        )
        perfil = perfil.join(ubicacion, how="left", validate="many_to_one")
    return perfil.rename_axis(COL_ESTACION).reset_index()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def filtrar_periodo(
    df: pd.DataFrame,
    inicio: str | None = None,
    fin: str | None = None,
    col: str = COL_PERIODO,
    col_fin_objetivo: str | None = None,
) -> pd.DataFrame:
    """Filtra filas con ``inicio <= col < fin + 1 día``; los extremos son fechas inclusivas.

    Se usa para separar entrenamiento, validación y prueba en orden temporal con
    las fechas de ``config.py``. Para P2, pasar
    ``col_fin_objetivo="objetivo_fin_periodo"`` y el último día de la partición
    como ``fin``. Solo se conservan ventanas cuyo cierre exclusivo sea menor o
    igual al inicio del día siguiente. Así se purgan las últimas dos horas para
    la etiqueta de tres horas. Las ventanas incompletas también se excluyen.

    Generar primero los rezagos sobre el historial y filtrar después: el contexto
    anterior al inicio de validación/prueba sí está disponible al predecir.
    """
    _validar_columnas(df, [col], "a filtrar")
    mascara = pd.Series(True, index=df.index)
    if inicio is not None:
        mascara &= df[col] >= pd.Timestamp(inicio)
    if fin is not None:
        corte = pd.Timestamp(fin) + pd.Timedelta(days=1)
        mascara &= df[col] < corte
    if col_fin_objetivo is not None:
        if fin is None:
            raise ValueError("Se requiere fin para comprobar el cierre de la ventana objetivo.")
        _validar_columnas(df, [col_fin_objetivo], "a filtrar")
        mascara &= df[col_fin_objetivo].notna() & (df[col_fin_objetivo] <= corte)
    return df.loc[mascara].copy()


def construir_features(
    viajes: pd.DataFrame,
    nomenclatura: pd.DataFrame,
    horizonte: int = 1,
    horizonte_riesgo: int = VENTANA_RIESGO_HORAS,
    umbral_riesgo: float = UMBRAL_RIESGO,
    festivos: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Encadena P1/P2 para predecir al inicio de t con datos anteriores a t.

    ``horizonte=1`` representa el desplazamiento mínimo hasta la hora terminada
    anterior. Esta entrega no admite mayor anticipación. ``horizonte_riesgo`` es
    la duración de la ventana futura de P2, no el desplazamiento de predictores.
    Los conteos contemporáneos y las columnas ``objetivo_`` no deben entrar en X.
    """
    _validar_entero_positivo(horizonte, "horizonte")
    if horizonte != 1:
        raise ValueError("Esta entrega admite solo horizonte=1: predicción al inicio de t.")
    _validar_nomenclatura(nomenclatura, [NOM_CAPACIDAD])
    panel = construir_panel_estacion_periodo(viajes, estaciones=nomenclatura[NOM_ID])
    panel = agregar_variables_calendario(panel, festivos=festivos)
    panel = agregar_rezagos(panel, horizonte=horizonte)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    logger.success(f"Panel con {len(panel):,} filas guardado en {output_path}")


if __name__ == "__main__":
    app()
