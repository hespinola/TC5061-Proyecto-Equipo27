# Evidencia de carga y recursos

| Usuarios | Solicitudes | Errores | RPS global | P95 (ms) | P99 (ms) |
|---:|---:|---:|---:|---:|---:|
| 10 | 1,545 | 0 | 12.91 | 32 | 54 |
| 30 | 4,400 | 0 | 36.79 | 160 | 290 |
| 60 | 4,874 | 0 | 40.73 | 960 | 1,100 |

| Usuarios | CPU API media de muestras (%) | Memoria máxima API (MiB) | Memoria máxima MLflow (MiB) |
|---:|---:|---:|---:|
| 10 | 12.00 | 215.0 | 1,010 |
| 30 | 51.53 | 220.0 | 1,021 |
| 60 | 93.99 | 230.4 | 1,022 |

Cada escenario tiene 32 muestras por contenedor, sin errores de captura.
La media de CPU es aritmética sobre esas muestras, no una integral de CPU-tiempo.
Al finalizar se verificó `RestartCount=0` y `OOMKilled=false` en ambos servicios.
Los contenedores se detuvieron conservando el volumen de MLflow.

Las carpetas `2026-11-22/u10`, `u30` y `u60` corresponden a ejecuciones
secuenciales de 120 segundos con 10, 30 y 60 usuarios, respectivamente.
Los timestamps internos están en UTC; la fecha de la carpeta es de México.

Condiciones de la medición:

- Windows con Docker Desktop/WSL2; anfitrión AMD Ryzen 7 9800X3D (8 núcleos,
  16 procesadores lógicos). Docker reporta 16 CPU y 16,323,862,528 bytes de RAM.
- API y MLflow limitados individualmente a 1 CPU y 1 GiB mediante Compose.
- Modelo P1, versión local 1, reconstruido en la entrega A/B; no es el modelo
  pequeño de demostración. Identidad y corrida en `ejecucion.json`.
- El modelo se exportó con Python 3.14.5; el contenedor lo cargó y sirvió con
  Python 3.13.18. MLflow advirtió diferencia de Python y cloudpickle
  (3.1.2 al guardar frente a 3.1.3 al cargar); no se atribuye identidad de
  entorno entre entrenamiento e inferencia.
- Antes de medir se verificó el flujo completo con 2 usuarios durante 15 s.
  Los contenedores permanecieron activos entre escenarios; no se midió arranque
  en frío. Cada escenario conserva su rampa y todas sus solicitudes.
- Peticiones con una sola instancia y payload fijo. El generador comparte el
  anfitrión con Docker: los resultados son locales, no una estimación directa
  de capacidad en una instancia cloud.

El HTML de Locust y las gráficas PNG documentan la parte 3. `recursos.csv` y
`stats_raw.jsonl` documentan la parte 4. Las mediciones se conservan sin eliminar
errores ni picos. Los percentiles globales se consultan en `locust_stats.csv`;
los de las gráficas temporales corresponden a las ventanas de Locust.

Las instrucciones reproducibles están en [load_tests/README.md](../../load_tests/README.md).
Estos archivos son insumos para el compañero responsable de FinOps y del
reporte integrado; aquí no se calculan costos ni se redacta ese reporte.
