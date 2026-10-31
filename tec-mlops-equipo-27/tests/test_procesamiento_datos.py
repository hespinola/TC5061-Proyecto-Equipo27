"""Procesamiento de datos (``mibici/dataset.py``): casos normales y casos límite.

Funciones bajo prueba: ``nombre_columna``, ``normalizar_viajes``, ``normalizar_estaciones``,
``version_manifest``, ``sha256_archivo`` y ``detectar_codificacion``.

No descargan nada ni requieren MLflow. Ejecución (desde ``tec-mlops-equipo-27/``)::

    python -m pytest tests/test_procesamiento_datos.py -v
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from mibici import dataset as d

INICIO = "2026-01-01"
FIN = "2026-02-01"  # cierre exclusivo


@pytest.fixture
def viajes_crudos() -> pd.DataFrame:
    """Tres viajes válidos con nombres de columna como los publica MiBici."""
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3],
            "Inicio_del_viaje": ["2026-01-01 00:00", "2026-01-15 12:30", "2026-01-31 23:59"],
            "Fin_del_viaje": ["2026-01-01 00:20", "2026-01-15 12:50", "2026-02-01 00:10"],
            "Origen_Id": [1, 2, 3],
            "Destino_Id": [2, 3, 1],
        }
    )


def un_viaje(**cambios) -> pd.DataFrame:
    """Un único viaje válido, con los campos que se indiquen reemplazados."""
    fila = {
        "Viaje_Id": 1,
        "Inicio_del_viaje": "2026-01-10 08:00",
        "Fin_del_viaje": "2026-01-10 08:20",
        "Origen_Id": 1,
        "Destino_Id": 2,
    }
    fila.update(cambios)
    return pd.DataFrame([fila])


# ---------------------------------------------------------------------------
# nombre_columna
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("original", "esperado"),
    [
        ("Inicio del Viaje", "inicio_del_viaje"),
        ("  Origen_Id ", "origen_id"),
        ("Género", "genero"),  # quita acentos
        ("Año de nacimiento", "ano_de_nacimiento"),
        ("FIN-VIAJE", "fin_viaje"),
        ("__a___b__", "a_b"),  # colapsa y recorta separadores
        ("", ""),  # límite: nombre vacío
        (123, "123"),  # límite: no es texto
    ],
)
def test_nombre_columna(original, esperado):
    assert d.nombre_columna(original) == esperado


# ---------------------------------------------------------------------------
# normalizar_viajes
# ---------------------------------------------------------------------------
def test_normalizar_viajes_caso_normal(viajes_crudos):
    salida, stats = d.normalizar_viajes(viajes_crudos, INICIO, FIN)

    assert list(salida.columns) == d.SCHEMA.names
    assert salida.viaje_id.tolist() == [1, 2, 3]
    assert stats == {"originales": 3, "invalidos": 0, "fuera_periodo": 0}
    for columna in ("viaje_id", "origen_id", "destino_id"):
        assert str(salida[columna].dtype) == "int64"
    assert pd.api.types.is_datetime64_any_dtype(salida.inicio_del_viaje)


def test_viaje_que_termina_despues_del_periodo_se_conserva(viajes_crudos):
    # El periodo se decide por el inicio del viaje: el tercero empieza el 31 y termina el 1.
    salida, _ = d.normalizar_viajes(viajes_crudos, INICIO, FIN)
    assert 3 in salida.viaje_id.tolist()


def test_campos_personales_no_pasan_a_la_salida(viajes_crudos):
    viajes_crudos["Usuario_Id"] = [10, 20, 30]
    viajes_crudos["Género"] = ["F", "M", "F"]
    viajes_crudos["Año_de_Nacimiento"] = [1990, 1985, 2000]

    salida, _ = d.normalizar_viajes(viajes_crudos, INICIO, FIN)

    assert list(salida.columns) == d.SCHEMA.names


@pytest.mark.parametrize(
    ("inicio_viaje", "dentro"),
    [
        ("2026-01-01 00:00:00", True),  # límite inferior inclusivo
        ("2026-01-31 23:59:59", True),
        ("2026-02-01 00:00:00", False),  # límite superior exclusivo
        ("2025-12-31 23:59:59", False),
    ],
)
def test_limites_del_periodo(inicio_viaje, dentro):
    df = un_viaje(Inicio_del_viaje=inicio_viaje, Fin_del_viaje="2026-02-02 00:00")

    salida, stats = d.normalizar_viajes(df, INICIO, FIN)

    assert len(salida) == int(dentro)
    assert stats["fuera_periodo"] == int(not dentro)
    assert stats["invalidos"] == 0


@pytest.mark.parametrize("id_invalido", [0, -5, 1.5, "abc", None, np.inf])
def test_ids_no_enteros_positivos_son_invalidos(id_invalido):
    for columna in ("Viaje_Id", "Origen_Id", "Destino_Id"):
        salida, stats = d.normalizar_viajes(un_viaje(**{columna: id_invalido}), INICIO, FIN)
        assert salida.empty, columna
        assert stats["invalidos"] == 1, columna


@pytest.mark.parametrize("campo", ["Inicio_del_viaje", "Fin_del_viaje"])
def test_fecha_ilegible_o_nula_es_invalida(campo):
    for valor in ("no es fecha", None):
        salida, stats = d.normalizar_viajes(un_viaje(**{campo: valor}), INICIO, FIN)
        assert salida.empty
        assert stats == {"originales": 1, "invalidos": 1, "fuera_periodo": 0}


def test_fin_anterior_al_inicio_es_invalido():
    df = un_viaje(Inicio_del_viaje="2026-01-10 08:00", Fin_del_viaje="2026-01-10 07:59")
    salida, stats = d.normalizar_viajes(df, INICIO, FIN)
    assert salida.empty
    assert stats["invalidos"] == 1


def test_viaje_de_duracion_cero_es_valido():
    df = un_viaje(Inicio_del_viaje="2026-01-10 08:00", Fin_del_viaje="2026-01-10 08:00")
    salida, stats = d.normalizar_viajes(df, INICIO, FIN)
    assert len(salida) == 1
    assert stats["invalidos"] == 0


def test_tabla_vacia_devuelve_salida_vacia_con_esquema():
    vacia = un_viaje().iloc[0:0]
    salida, stats = d.normalizar_viajes(vacia, INICIO, FIN)
    assert salida.empty
    assert list(salida.columns) == d.SCHEMA.names
    assert stats == {"originales": 0, "invalidos": 0, "fuera_periodo": 0}


def test_columna_operativa_faltante_levanta_error():
    with pytest.raises(ValueError, match="Faltan columnas operativas"):
        d.normalizar_viajes(un_viaje().drop(columns="Destino_Id"), INICIO, FIN)


def test_columnas_que_chocan_tras_normalizar_levantan_error():
    df = un_viaje()
    df["ID Viaje"] = 99  # alias de viaje_id: queda duplicada tras normalizar
    with pytest.raises(ValueError, match="duplicadas"):
        d.normalizar_viajes(df, INICIO, FIN)


def test_fechas_con_zona_horaria_se_rechazan():
    df = un_viaje(
        Inicio_del_viaje="2026-01-10T08:00:00+00:00", Fin_del_viaje="2026-01-10T08:20:00+00:00"
    )
    with pytest.raises(ValueError, match="zona horaria"):
        d.normalizar_viajes(df, INICIO, FIN)


# ---------------------------------------------------------------------------
# normalizar_estaciones
# ---------------------------------------------------------------------------
@pytest.fixture
def nomenclatura_cruda() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [30, 10, 20],
            "latitude": [20.67, 20.68, 20.69],
            "longitude": [-103.35, -103.36, -103.37],
            "DPCapacity": [20, 12, 40],
            "Name": ["C", "A", "B"],
        }
    )


def test_normalizar_estaciones_caso_normal(nomenclatura_cruda):
    salida = d.normalizar_estaciones(nomenclatura_cruda)

    assert salida.id.tolist() == [10, 20, 30]  # ordenada por id
    assert salida.name.tolist() == ["A", "B", "C"]  # conserva el nombre si existe
    assert salida.dpcapacity.tolist() == [12, 40, 20]
    assert str(salida.id.dtype) == "int64"


def test_normalizar_estaciones_sin_columna_de_nombre(nomenclatura_cruda):
    salida = d.normalizar_estaciones(nomenclatura_cruda.drop(columns="Name"))
    assert "name" not in salida.columns


@pytest.mark.parametrize("capacidad", [0, -3, np.inf, "n/d", None])
def test_capacidad_invalida_queda_en_na(nomenclatura_cruda, capacidad):
    nomenclatura_cruda["DPCapacity"] = [20, capacidad, 40]  # ids 30, 10, 20
    salida = d.normalizar_estaciones(nomenclatura_cruda).set_index("id")

    assert pd.isna(salida.loc[10, "dpcapacity"])  # solo la capacidad inválida se anula
    assert salida.loc[30, "dpcapacity"] == 20
    assert salida.loc[20, "dpcapacity"] == 40


@pytest.mark.parametrize(
    ("latitud", "longitud", "lat_valida", "lon_valida"),
    [
        (20.67, -103.35, True, True),
        (90, 180, True, True),  # límites inclusivos
        (-90, -180, True, True),
        (90.0001, -103.35, False, True),
        (20.67, -180.5, True, False),
    ],
)
def test_coordenadas_fuera_de_rango_quedan_en_na(
    nomenclatura_cruda, latitud, longitud, lat_valida, lon_valida
):
    nomenclatura_cruda.loc[0, ["latitude", "longitude"]] = [latitud, longitud]
    fila = d.normalizar_estaciones(nomenclatura_cruda).set_index("id").loc[30]

    assert pd.notna(fila.latitude) == lat_valida
    assert pd.notna(fila.longitude) == lon_valida


@pytest.mark.parametrize("id_malo", [0, -1, 1.5, None, "x"])
def test_ids_invalidos_levantan_error(nomenclatura_cruda, id_malo):
    nomenclatura_cruda["ID"] = pd.Series([id_malo, 10, 20], dtype="object")
    with pytest.raises(ValueError, match="identificadores inválidos o duplicados"):
        d.normalizar_estaciones(nomenclatura_cruda)


def test_ids_duplicados_levantan_error(nomenclatura_cruda):
    nomenclatura_cruda.loc[0, "ID"] = 10
    with pytest.raises(ValueError, match="identificadores inválidos o duplicados"):
        d.normalizar_estaciones(nomenclatura_cruda)


def test_columna_requerida_faltante_levanta_error(nomenclatura_cruda):
    with pytest.raises(ValueError, match="requiere columnas únicas"):
        d.normalizar_estaciones(nomenclatura_cruda.drop(columns="DPCapacity"))


# ---------------------------------------------------------------------------
# version_manifest
# ---------------------------------------------------------------------------
def test_version_manifest_es_un_sha256_determinista():
    manifiesto = {"inicio": "2025-10-01", "archivos": {"2025-10": {"sha256": "abc"}}}
    version = d.version_manifest(manifiesto)
    assert len(version) == 64
    assert int(version, 16) >= 0
    assert d.version_manifest(manifiesto) == version


def test_version_manifest_ignora_data_version_y_el_orden_de_llaves():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1, "data_version": "lo-que-sea"}
    assert d.version_manifest(a) == d.version_manifest(b)


@pytest.mark.parametrize("cambio", [{"x": 2}, {"y": {"a": 3, "b": 99}}, {"z": None}])
def test_version_manifest_cambia_si_cambia_el_contenido(cambio):
    base = {"x": 1, "y": {"a": 3, "b": 2}}
    assert d.version_manifest({**base, **cambio}) != d.version_manifest(base)


# ---------------------------------------------------------------------------
# sha256_archivo y detectar_codificacion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("contenido", "hash_esperado"),
    [
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), # pragma: allowlist secret
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),  # pragma: allowlist secret
    ],
)
def test_sha256_archivo_valores_conocidos(tmp_path, contenido, hash_esperado):
    archivo = tmp_path / "datos.bin"
    archivo.write_bytes(contenido)
    assert d.sha256_archivo(archivo) == hash_esperado


def test_sha256_archivo_lee_por_bloques_sin_alterar_el_resultado(tmp_path):
    contenido = b"0123456789" * 300_000  # ~2.9 MiB: más de dos bloques de 1 MiB
    archivo = tmp_path / "grande.bin"
    archivo.write_bytes(contenido)
    assert d.sha256_archivo(archivo) == hashlib.sha256(contenido).hexdigest()


@pytest.mark.parametrize(
    ("contenido", "codificacion"),
    [
        ("nombre\nEstación céntrica\n".encode("utf-8-sig"), "utf-8-sig"),
        ("nombre\nEstación céntrica\n".encode("cp1252"), "cp1252"),
        (b"", "utf-8-sig"),  # límite: archivo vacío
        (b"abc\x81\n", "latin-1"),  # 0x81 no existe en cp1252: cae al último respaldo
    ],
)
def test_detectar_codificacion(tmp_path, contenido, codificacion):
    archivo = tmp_path / "datos.csv"
    archivo.write_bytes(contenido)
    assert d.detectar_codificacion(archivo) == codificacion
