"""Validación de calidad e integridad del panel estación x hora con DeepChecks.

Se ejecuta después de ``make features`` y antes de entrenar:

    python -m mibici.validacion

Revisa dos cosas sobre ``panel_estacion_hora.parquet``:

* **Integridad** (entrenamiento y prueba por separado): valores faltantes, nulos
  mezclados, tipos de dato mezclados, columnas constantes, claves
  ``estación x hora`` duplicadas y fuga de información (una variable que por sí
  sola predice el objetivo).
* **Entrenamiento vs. prueba**: drift de las variables de demanda y de los
  objetivos de P1 y P2, estaciones nuevas en prueba y traslape de fechas entre
  particiones (la partición es temporal: prueba debe ser posterior).

Cada chequeo tiene una condición con umbral explícito (``UMBRALES``). El comando
guarda un resumen JSON versionable y el reporte HTML completo de DeepChecks.
Por defecto termina con código 0 aunque haya hallazgos, porque su función es
documentarlos; con ``--estricto`` termina con código 1 si alguna condición falla.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici import features as f
from mibici.config import (
    FEATURES_P2,
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

app = typer.Typer()

ETIQUETA_P1 = "salidas"
ETIQUETA_P2 = "objetivo_en_riesgo"
COL_CATEGORICAS = (f.COL_ESTACION,)
FEATURES = (f.COL_ESTACION, *FEATURES_P2)
# ``mes`` y las demás variables de calendario cambian por construcción entre
# particiones temporales (prueba cubre meses que entrenamiento no ve), así que su
# drift no indica un problema de datos. Se mide drift solo en la demanda observada.
FEATURES_DRIFT = tuple(
    c for c in FEATURES_P2 if c.startswith(("salidas_", "llegadas_")) or c == "capacidad"
)

UMBRALES = {
    # Proporción máxima de nulos por columna. Los rezagos de 168 h son nulos
    # durante la primera semana de cada estación (~2 % de un año); 5 % deja margen.
    "nulos_max": 0.05,
    # Ninguna clave estación x hora puede repetirse.
    "duplicados_max": 0.0,
    # Predictive Power Score máximo de una sola variable sobre el objetivo; uno
    # mayor sugiere fuga (por ejemplo, el conteo de la hora actual como predictor).
    "pps_max": 0.8,
    # Distancia de drift máxima (Kolmogorov-Smirnov / Cramér's V).
    "drift_max": 0.2,
    # Proporción máxima de filas de prueba con estaciones que no aparecen en
    # entrenamiento.
    "categorias_nuevas_max": 0.01,
}

RESUMEN_PATH = REPORTS_DIR / "validacion_datos.json"
HTML_DIR = REPORTS_DIR / "deepchecks"


def preparar_deepchecks():
    """Importa DeepChecks 0.19.1 sobre NumPy 2 y scikit-learn 1.9.

    DeepChecks 0.19.1 es la última versión publicada y fue escrita para NumPy 1 y
    scikit-learn < 1.6. Para no degradar el entorno del proyecto se restauran dos
    nombres que esas bibliotecas retiraron, solo si faltan:

    * ``np.Inf`` y ``np.NaN`` (alias de ``np.inf`` y ``np.nan``).
    * El scorer ``max_error``, que scikit-learn ahora registra como ``neg_max_error``.

    Los chequeos elegidos en este módulo no dependen de modelos ni de scorers; los
    que fallan con pandas 3 (``OutlierSampleDetection``, ``TrainTestSamplesMix`` y
    ``MultivariateDrift``) no se usan. La función es idempotente.
    """
    for viejo, nuevo in (("Inf", np.inf), ("NaN", np.nan)):
        if viejo not in np.__dict__:
            setattr(np, viejo, nuevo)

    from sklearn.metrics import _scorer

    scorers = _scorer._SCORERS
    if "max_error" not in scorers and "neg_max_error" in scorers:
        scorers["max_error"] = scorers["neg_max_error"]

    import deepchecks.tabular as dc_tabular
    import deepchecks.tabular.checks  # noqa: F401  (registra dc_tabular.checks)

    return dc_tabular


@dataclass(frozen=True)
class Particiones:
    entrenamiento: pd.DataFrame
    prueba: pd.DataFrame


def particionar(panel: pd.DataFrame) -> Particiones:
    """Separa entrenamiento y prueba con las mismas fechas que el modelado.

    Usa ``filtrar_periodo`` con la ventana objetivo de P2 para que las etiquetas no
    crucen el corte. Validación queda fuera: aquí se compara el extremo anterior
    (entrenamiento) con el posterior (prueba) para detectar el mayor drift posible.
    """
    inicio_prueba = (pd.Timestamp(FIN_VALIDACION) + pd.Timedelta(days=1)).date().isoformat()
    entrenamiento = f.filtrar_periodo(
        panel, fin=FIN_ENTRENAMIENTO, col_fin_objetivo="objetivo_fin_periodo"
    )
    fin_prueba = panel[f.COL_PERIODO].max().date().isoformat()
    prueba = f.filtrar_periodo(
        panel,
        inicio=inicio_prueba,
        fin=fin_prueba,
        col_fin_objetivo="objetivo_fin_periodo",
    )
    if entrenamiento.empty or prueba.empty:
        raise ValueError(
            f"Particiones vacías: entrenamiento={len(entrenamiento)}, prueba={len(prueba)}."
        )
    return Particiones(entrenamiento, prueba)


def _dataset(dc, df: pd.DataFrame, etiqueta: str, nombre: str):
    columnas = [f.COL_PERIODO, *FEATURES, etiqueta]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise KeyError(f"Al panel le faltan las columnas {faltantes}.")
    datos = df[columnas].copy()
    if etiqueta == ETIQUETA_P2:
        # Las filas sin capacidad válida no tienen etiqueta de P2.
        datos = datos.loc[datos[etiqueta].notna()]
        datos[etiqueta] = datos[etiqueta].astype("int8")
        tipo = "binary"
    else:
        tipo = "regression"
    return dc.Dataset(
        datos,
        label=etiqueta,
        features=list(FEATURES),
        cat_features=list(COL_CATEGORICAS),
        datetime_name=f.COL_PERIODO,
        label_type=tipo,
        dataset_name=nombre,
    )


def suite_integridad(dc, umbrales: dict = UMBRALES):
    c = dc.checks
    return dc.Suite(
        "Integridad del panel",
        c.PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(umbrales["nulos_max"]),
        c.MixedNulls().add_condition_different_nulls_less_equal_to(1),
        c.MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.01, 0.1)),
        c.IsSingleValue().add_condition_not_single_value(),
        c.DataDuplicates(
            columns=[f.COL_ESTACION, f.COL_PERIODO], n_samples=None
        ).add_condition_ratio_less_or_equal(umbrales["duplicados_max"]),
        c.FeatureLabelCorrelation().add_condition_feature_pps_less_than(umbrales["pps_max"]),
    )


def suite_entrenamiento_prueba(dc, umbrales: dict = UMBRALES):
    c = dc.checks
    return dc.Suite(
        "Entrenamiento vs. prueba",
        c.DatasetsSizeComparison(),
        c.DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
        c.NewCategoryTrainTest().add_condition_new_categories_less_or_equal(
            umbrales["categorias_nuevas_max"]
        ),
        c.FeatureDrift(columns=list(FEATURES_DRIFT)).add_condition_drift_score_less_than(
            max_allowed_categorical_score=umbrales["drift_max"],
            max_allowed_numeric_score=umbrales["drift_max"],
        ),
        c.LabelDrift().add_condition_drift_score_less_than(umbrales["drift_max"]),
    )


def suite_objetivo_p2(dc, umbrales: dict = UMBRALES):
    return dc.Suite(
        "Drift del objetivo de P2",
        dc.checks.LabelDrift().add_condition_drift_score_less_than(umbrales["drift_max"]),
    )


def _resumir(resultado) -> list[dict]:
    filas = []
    for r in resultado.results:
        error = getattr(r, "exception", None)
        if error is not None:
            filas.append({"chequeo": r.get_header(), "estado": "error", "detalle": str(error)})
            continue
        condiciones = r.conditions_results
        estado = "ok" if all(c.is_pass() for c in condiciones) else "falla"
        detalle = "; ".join(c.details for c in condiciones if c.details)
        filas.append({"chequeo": r.get_header(), "estado": estado, "detalle": detalle})
    return filas


def validar(panel: pd.DataFrame, html_dir: Path | None = None, umbrales: dict = UMBRALES) -> dict:
    """Corre los chequeos sobre el panel y devuelve un resumen serializable."""
    dc = preparar_deepchecks()
    # Solo las columnas que se validan: evita copiar el panel completo al particionar.
    columnas = [f.COL_PERIODO, *FEATURES, ETIQUETA_P1, ETIQUETA_P2, "objetivo_fin_periodo"]
    partes = particionar(panel[[c for c in dict.fromkeys(columnas) if c in panel.columns]])
    corridas = {
        "integridad_entrenamiento": (
            suite_integridad(dc, umbrales),
            (_dataset(dc, partes.entrenamiento, ETIQUETA_P1, "entrenamiento"),),
        ),
        "integridad_prueba": (
            suite_integridad(dc, umbrales),
            (_dataset(dc, partes.prueba, ETIQUETA_P1, "prueba"),),
        ),
        "entrenamiento_vs_prueba_p1": (
            suite_entrenamiento_prueba(dc, umbrales),
            (
                _dataset(dc, partes.entrenamiento, ETIQUETA_P1, "entrenamiento"),
                _dataset(dc, partes.prueba, ETIQUETA_P1, "prueba"),
            ),
        ),
        "entrenamiento_vs_prueba_p2": (
            suite_objetivo_p2(dc, umbrales),
            (
                _dataset(dc, partes.entrenamiento, ETIQUETA_P2, "entrenamiento"),
                _dataset(dc, partes.prueba, ETIQUETA_P2, "prueba"),
            ),
        ),
    }
    resumen = {
        "filas": {"entrenamiento": len(partes.entrenamiento), "prueba": len(partes.prueba)},
        "umbrales": umbrales,
        "suites": {},
    }
    for nombre, (suite, datasets) in corridas.items():
        resultado = suite.run(*datasets, with_display=html_dir is not None)
        resumen["suites"][nombre] = _resumir(resultado)
        if html_dir is not None:
            html_dir.mkdir(parents=True, exist_ok=True)
            resultado.save_as_html(str(html_dir / f"{nombre}.html"), as_widget=False)
    estados = [c["estado"] for filas in resumen["suites"].values() for c in filas]
    resumen["aprobado"] = all(e == "ok" for e in estados)
    return resumen


def imprimir(resumen: dict) -> None:
    for suite, filas in resumen["suites"].items():
        print(f"\n== {suite}")
        for fila in filas:
            marca = {"ok": "OK   ", "falla": "FALLA", "error": "ERROR"}[fila["estado"]]
            detalle = f"  {fila['detalle']}" if fila["detalle"] else ""
            print(f"  [{marca}] {fila['chequeo']}{detalle}")
    total = "sin hallazgos" if resumen["aprobado"] else "con hallazgos documentados"
    print(f"\nValidación {total}. Filas: {resumen['filas']}")


@app.command()
def main(
    panel_path: Path = PROCESSED_DATA_DIR / "panel_estacion_hora.parquet",
    resumen_path: Path = RESUMEN_PATH,
    html_dir: Path = HTML_DIR,
    estricto: bool = typer.Option(False, help="Código de salida 1 si alguna condición falla."),
):
    """Valida el panel con DeepChecks y guarda el resumen JSON y los reportes HTML."""
    logger.info(f"Leyendo {panel_path}")
    panel = pd.read_parquet(panel_path)
    resumen = validar(panel, html_dir=html_dir)
    resumen_path.parent.mkdir(parents=True, exist_ok=True)
    resumen_path.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n")
    imprimir(resumen)
    logger.success(f"Resumen en {resumen_path}; reportes HTML en {html_dir}")
    if estricto and not resumen["aprobado"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
