"""Contratos de ingesta y versionado con archivos sintéticos en carpetas temporales."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from mibici import dataset as d
from mibici.preparar import preparar, verificar_datos


def viajes():
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3, 4],
            "Inicio_del_viaje": [
                "2026-01-01 01:00",
                "2026-01-01 02:00",
                "mal",
                "2026-02-01 00:00",
            ],
            "Fin_del_viaje": ["2026-01-01 01:20", "2026-01-01 02:20", "mal", "2026-02-01 00:20"],
            "Origen_Id": [1, 2, 1, 1],
            "Destino_Id": [2, 1, 2, 2],
            "Usuario_Id": [100, 200, 300, 400],
            "Género": ["M", "F", "M", "F"],
        }
    )


class NormalizacionTests(unittest.TestCase):
    def test_limpieza_tipos_periodo_y_exclusion_de_campos_personales(self):
        out, stats = d.normalizar_viajes(viajes(), "2026-01-01", "2026-02-01")
        self.assertEqual(out.viaje_id.tolist(), [1, 2])
        self.assertEqual(stats, {"originales": 4, "invalidos": 1, "fuera_periodo": 1})
        self.assertEqual(list(out), d.SCHEMA.names)
        self.assertEqual(str(out.origen_id.dtype), "int64")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out.inicio_del_viaje))

    def test_columnas_con_acentos_y_alias(self):
        df = viajes().rename(columns={"Viaje_Id": " ID Viaje ", "Origen_Id": " Orígen Id "})
        out, _ = d.normalizar_viajes(df, "2026-01-01", "2026-02-01")
        self.assertEqual(len(out), 2)
        with self.assertRaises(ValueError):
            d.normalizar_viajes(df.drop(columns="Destino_Id"), "2026-01-01", "2026-02-01")

    def test_codificacion_estricta_no_corrompe_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            for encoding in ("utf-8-sig", "cp1252"):
                path = Path(tmp) / "datos.csv"
                path.write_bytes("nombre\nEstación céntrica\n".encode(encoding))
                elegido = d.detectar_codificacion(path)
                self.assertIn("Estación céntrica", path.read_text(encoding=elegido))

    def test_nomenclatura_rechaza_duplicados_y_marca_capacidades(self):
        nom = pd.DataFrame(
            {"id": [1, 2], "latitude": [20, 300], "longitude": [-103, -103], "dpcapacity": [0, 20]}
        )
        out = d.normalizar_estaciones(nom)
        self.assertTrue(pd.isna(out.iloc[0].dpcapacity))
        self.assertTrue(pd.isna(out.iloc[1].latitude))
        with self.assertRaises(ValueError):
            d.normalizar_estaciones(pd.concat([nom, nom.iloc[[0]]]))


class DescargaTests(unittest.TestCase):
    @staticmethod
    def respuesta(status, contenido=b"id\n1\n", content_type="text/csv"):
        response = requests.Response()
        response.status_code = status
        response.url = "https://ejemplo.invalid/viajes.csv"
        response.headers["Content-Type"] = content_type
        response._content = contenido
        response._content_consumed = True
        return response

    def test_descarga_envia_cabeceras_y_conserva_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viajes.csv"
            session = Mock()
            session.get.return_value = self.respuesta(200)
            d.descargar("https://ejemplo.invalid/viajes.csv", path, None, session)
            self.assertEqual(path.read_bytes(), b"id\n1\n")
            self.assertEqual(session.get.call_args.kwargs["headers"], d.CABECERAS_DESCARGA)
            self.assertEqual(session.get.call_count, 1)

    def test_406_visita_portal_y_reintenta_una_vez(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viajes.csv"
            session = Mock()
            session.get.side_effect = [
                self.respuesta(406),
                self.respuesta(200, b"<html>Portal</html>", "text/html"),
                self.respuesta(200),
            ]
            d.descargar("https://ejemplo.invalid/viajes.csv", path, None, session)
            self.assertEqual(path.read_bytes(), b"id\n1\n")
            self.assertEqual(session.get.call_count, 3)
            self.assertEqual(session.get.call_args_list[1].args[0], d.PORTAL_DATOS_URL)

    def test_406_persistente_no_crea_archivo_ni_reintenta_indefinidamente(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viajes.csv"
            session = Mock()
            session.get.side_effect = [
                self.respuesta(406),
                self.respuesta(200),
                self.respuesta(406),
            ]
            with self.assertRaisesRegex(requests.HTTPError, "mantiene HTTP 406"):
                d.descargar("https://ejemplo.invalid/viajes.csv", path, None, session)
            self.assertEqual(session.get.call_count, 3)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".csv.part").exists())

    def test_rechaza_html_o_archivo_vacio_sin_sobrescribir_original(self):
        for contenido, tipo in (
            (b"<html>Error</html>", "text/html"),
            (b"<!doctype html><html>Error</html>", "text/plain"),
            (b"", "text/csv"),
        ):
            with self.subTest(tipo=tipo), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "viajes.csv"
                path.write_bytes(b"original")
                session = Mock()
                session.get.return_value = self.respuesta(200, contenido, tipo)
                with self.assertRaises(ValueError):
                    d.descargar(
                        "https://ejemplo.invalid/viajes.csv", path, None, session, forzar=True
                    )
                self.assertEqual(path.read_bytes(), b"original")
                self.assertFalse(path.with_suffix(".csv.part").exists())


class ManifiestoTests(unittest.TestCase):
    def preparar(self, root):
        raw = root / "raw"
        raw.mkdir()
        df = pd.concat([viajes(), viajes().iloc[[0]]], ignore_index=True)
        df.to_csv(raw / "datos_abiertos_2026_01.csv", index=False)
        pd.DataFrame(
            {
                "id": [1, 2],
                "latitude": [20.6, 20.7],
                "longitude": [-103, -103],
                "dpcapacity": [20, 30],
            }
        ).to_csv(raw / "nomenclatura.csv", index=False)
        return {
            "raw_dir": raw,
            "processed_dir": root / "processed",
            "manifest_path": root / "references" / "data_manifest.json",
            "meses": ["2026-01"],
            "urls": {"2026-01": "https://ejemplo.invalid/viajes.csv"},
            "nomenclatura_url": "https://ejemplo.invalid/estaciones.csv",
            "inicio": "2026-01-01",
            "fin": "2026-02-01",
        }

    def test_reproduce_version_y_deduplica_entre_bloques_sin_red(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "CHUNK_SIZE", 2):
            kwargs = self.preparar(Path(tmp))
            with patch.object(
                d.requests.Session, "get", side_effect=AssertionError("No debe descargar")
            ):
                a = d.ingestar(**kwargs)
                b = d.ingestar(**kwargs)
            self.assertEqual(a, b)
            self.assertEqual(a["resumen"]["validos"], 2)
            self.assertEqual(a["resumen"]["duplicados"], 1)
            verificar_datos(kwargs["processed_dir"], kwargs["manifest_path"])
            self.assertEqual(len(pd.read_parquet(kwargs["processed_dir"] / "viajes.parquet")), 2)

    def test_detecta_cambios_en_fuente_y_no_sobrescribe_manifiesto(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self.preparar(Path(tmp))
            original = d.ingestar(**kwargs)
            path = kwargs["raw_dir"] / "datos_abiertos_2026_01.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Hash distinto"):
                d.ingestar(**kwargs)
            self.assertEqual(json.loads(kwargs["manifest_path"].read_text()), original)

    def test_panel_respeta_cobertura_y_rechaza_datos_limpios_alterados(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self.preparar(Path(tmp))
            d.ingestar(**kwargs)
            panel = preparar(kwargs["processed_dir"], kwargs["manifest_path"])
            self.assertEqual(len(panel), 2 * 31 * 24)
            self.assertEqual(panel.salidas.sum(), 2)
            self.assertEqual(panel.llegadas.sum(), 2)
            self.assertEqual(panel.periodo.max(), pd.Timestamp("2026-01-31 23:00"))
            self.assertTrue(panel.groupby("estacion_id").tail(2).objetivo_en_riesgo.isna().all())
            path = kwargs["processed_dir"] / "viajes.parquet"
            path.write_bytes(path.read_bytes() + b"alterado")
            with self.assertRaisesRegex(ValueError, "no coincide"):
                verificar_datos(kwargs["processed_dir"], kwargs["manifest_path"])

    def test_rechaza_id_repetido_con_datos_contradictorios(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self.preparar(Path(tmp))
            path = kwargs["raw_dir"] / "datos_abiertos_2026_01.csv"
            df = pd.read_csv(path)
            df.loc[len(df) - 1, "Destino_Id"] = 3
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "contradictorio"):
                d.ingestar(**kwargs)
            self.assertFalse(kwargs["manifest_path"].exists())


if __name__ == "__main__":
    unittest.main()
