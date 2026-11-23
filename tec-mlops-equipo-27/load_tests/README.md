# Partes 3 y 4: carga y recursos

Ejecutar desde `tec-mlops-equipo-27/`. Requiere Docker Desktop iniciado
(contenedores Linux), los servicios de `docker-compose.yml` y Python 3.13.
Instalar las herramientas en un entorno separado evita modificar las dependencias
de producción:

```powershell
py -3.13 -m venv .cache/load-venv
.cache/load-venv/Scripts/python.exe -m pip install -r requirements-load.txt
docker compose up --build -d --wait
.cache/load-venv/Scripts/python.exe load_tests/ejecutar.py --usuarios 10 --tasa 2 --segundos 120 --destino reports/carga/u10
.cache/load-venv/Scripts/python.exe load_tests/ejecutar.py --usuarios 30 --tasa 5 --segundos 120 --destino reports/carga/u30
.cache/load-venv/Scripts/python.exe load_tests/ejecutar.py --usuarios 60 --tasa 10 --segundos 120 --destino reports/carga/u60
docker compose stop
```

En Linux usar `python3 -m venv` y `.cache/load-venv/bin/python`. Antes de construir,
preparar `modelo_p1/` como indica el README principal. Para mediciones del modelo
real, exportar su versión exacta desde el registry; `make up` puede generar un
modelo **demo** si falta el artefacto. No mezclar mediciones de ambos modelos.
El healthcheck y los metadatos guardados permiten identificar cuál se utilizó.

El runner también ofrece `--motor podman` (requiere los mismos nombres de
contenedores), pero la ejecución documentada se valida con Docker. `--host`
solo admite localhost; `--intervalo` configura la pausa entre muestreos (2 s
por defecto, más el tiempo empleado por `stats`). No sustituye ni detiene servicios.
Rechaza destinos existentes para conservar evidencia de ejecuciones previas.

## Escenario y alcance

- Usuarios concurrentes de Locust, cada uno con espera aleatoria de 0.5–1 s
  después de cada respuesta: modelo de carga cerrado, no RPS impuesto.
- Cada POST a `/predict/demanda` usa `examples/predict-demanda.json`: una
  estación/hora y una inferencia por petición, sin variación de payload.
- Timeout HTTP de 30 s. Un 200 también se valida: una predicción finita,
  no negativa y de la estación solicitada; respuestas inválidas cuentan como errores.
- Cada escenario incluye la rampa de usuarios. No se excluye calentamiento;
  no es una prueba de capacidad máxima ni establece un SLO.
- El generador y Docker Desktop comparten anfitrión. La VM de Docker, otros
  procesos y límites de CPU/memoria influyen en la medición.
- Se muestrean API y MLflow **mientras Locust está activo**. CPU es el porcentaje
  que reporta el motor (puede superar 100% con varios núcleos); memoria se convierte
  a bytes. Los límites reservados no equivalen al consumo observado.
- MLflow puede consumir poco CPU porque la API sirve el modelo empaquetado;
  no consulta el registry por cada inferencia.

## Evidencia para las partes 5 y 6

Cada carpeta de ejecución contiene:

| Archivo | Contenido |
|---|---|
| `locust.html` | Reporte generado por Locust |
| `locust_stats.csv` | Totales, fallos, RPS y percentiles globales |
| `locust_stats_history.csv` | Serie temporal, usuarios y métricas por ventana |
| `locust_failures.csv`, `locust_exceptions.csv` | Fallos HTTP/de contrato y excepciones |
| `recursos.csv` | CPU, memoria, límite y tiempo por contenedor |
| `stats_raw.jsonl` | Salida original de stats y timestamps UTC |
| `ejecucion.json` | Parámetros, modelo, versiones, imagen, límites, código y errores de captura |
| `desempeno.png`, `recursos.png` | Gráficas listas para incorporar al reporte |

El log local de Locust queda excluido por `.gitignore`. Los CSV de esta carpeta
sí pueden versionarse: contienen métricas técnicas, no viajes ni datos personales.
El runner devuelve error si Locust detecta fallos o si la captura de recursos
resulta incompleta, y conserva la evidencia para investigarlos.

Para FinOps, utilizar **solicitudes exitosas** como inferencias exitosas en este
escenario de una fila por petición y declarar la ventana temporal utilizada.
Las series de stats incluyen el arranque de Locust y su cierre; al cruzarlas con
la carga usar los timestamps de `locust_stats_history.csv`. El costo, la selección
de una instancia y las conclusiones del reporte quedan a cargo de las partes 5 y 6.

Pruebas del procesamiento de métricas (con pytest del proyecto):

```bash
python -m pytest tests/test_carga_recursos.py -q
```

Referencia: [ejecución headless de Locust](https://docs.locust.io/en/stable/running-without-web-ui.html).
