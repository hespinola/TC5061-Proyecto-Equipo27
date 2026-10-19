# Ejecución y evidencias de la primera entrega

Estado: **ingesta reproducida y panel generado por el equipo; baselines pendientes**.
Los conteos siguientes provienen de las ejecuciones reportadas y del manifiesto
real compartido el 18 de octubre de 2026; no son datos de pruebas sintéticas.

| Verificación | Resultado |
| --- | --- |
| Periodo | 2025-10-01 a 2026-10-01, cierre exclusivo |
| Archivos mensuales | 12 |
| Viajes originales | 4,497,877 |
| Inválidos / fuera del mes | 0 / 0 |
| Duplicados eliminados | 758; contabilizados al procesar septiembre de 2026 |
| Viajes válidos | 4,497,119 |
| Estaciones observadas | 471 |
| Estaciones de la nomenclatura | 484 |
| IDs observados sin nomenclatura | 405 y 407 |
| Capacidades inválidas en el catálogo completo | 4 |
| Estaciones sin capacidad válida dentro del panel | 2, según la advertencia de features |
| Panel horario | 4,125,960 filas = 471 estaciones × 8,760 horas |
| Llegadas posteriores al cierre | 16; se conservan los viajes, no se extiende la observación |
| Codificación detectada | cp1252 en enero de 2026; utf-8-sig en los demás meses |
| Reproducción | Segunda ingesta con la misma versión y sin discrepancias de hashes |

Versión completa:
`ea4e65629d4c5f0565956bdb37c17920b8529a7a1308fd6ba7ecb74429f231d1`.

Se reportaron 27 pruebas satisfactorias y `pip check` sin conflictos antes del
ajuste de transporte HTTP 406. Confirmar también la suite ampliada de descargas
antes del commit del código que se utilizará para los experimentos.

## 1. Preparar el entorno y revisar el código

Desde la raíz de este repositorio, entrar al proyecto generado:

```bash
cd tec-mlops-equipo-27
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m ruff check --fix mibici/dataset.py mibici/preparar.py mibici/modeling/train.py tests/test_dataset.py tests/test_baselines.py
python -m ruff format mibici/config.py mibici/dataset.py mibici/preparar.py mibici/modeling/train.py tests/test_dataset.py tests/test_baselines.py
make test
```

Estas pruebas usan datos sintéticos locales y no descargan archivos de MiBici.
Si falla un comando, conservar la salida completa y corregir antes de continuar.
Las instrucciones de ejecución por sí solas no sustituyen sus salidas; el estado
verificado de cada etapa se registra al inicio de este documento.

## 2. Ingesta real y reproducción

```bash
make data
make data
```

La primera ejecución generará `references/data_manifest.json`. La segunda deberá
verificar los mismos hashes, conteos y versión sin descargas adicionales.
Guardar la salida de ambas. Revisar los conteos de originales, inválidos, duplicados,
válidos y estaciones. Si falta una columna, la ingesta debe detenerse; no ocultar
errores cambiando los nombres sin revisar el contrato.

`--actualizar-version` vuelve a descargar fuentes y acepta una nueva versión de
forma explícita. No utilizarlo para ocultar un fallo al reproducir la versión fijada.
El respaldo de originales fuera de Git todavía debe acordarse y documentarse.

## 3. Registrar el código y ejecutar baselines

Revisar primero el diff y confirmar en Git únicamente código, pruebas y manifiesto.
No agregar `data/`, archivos de MLflow ni modelos.

El comando de baselines exige código confirmado para que la etiqueta `git_commit`
identifique la implementación ejecutada. Una vez hecho ese commit:

```bash
make baseline
make tracking
```

El primer comando reconstruye el panel desde los datos verificados, ejecuta dos
referencias de P1 y un baseline por P2/P3, y guarda `reports/baseline_results.json`
y figuras. El segundo deja abierto el servidor de tracking en
http://127.0.0.1:5000; se detiene con Ctrl+C.

Si se configuró `MLFLOW_TRACKING_URI` para otro servidor, abrir ese servidor en
lugar del tracking local. El objetivo es que UI y entrenamiento consulten el mismo
almacén, no crear experimentos en bases diferentes.

## 4. Evidencias de la interfaz de MLflow

Guardar en `reports/figures/` cuatro capturas reales, legibles:

1. `mlflow_01_runs.png`: experimento `mibici-equipo27` y lista de runs P1/P2/P3.
2. `mlflow_02_detalle.png`: un run con parámetros, métricas y etiquetas de equipo,
   pregunta y commit; expandir secciones o ajustar zoom para que se lean.
3. `mlflow_03_artefacto.png`: figura registrada dentro de un run.
4. `mlflow_04_comparacion.png`: comparar las referencias semanal/diaria de P1,
   con la misma métrica y partición.

Los PNG generados por matplotlib son artefactos válidos, pero no sustituyen estas
capturas de la interfaz exigidas por la actividad.

## 5. Documentación, propuesta y colaboración

- Guardar el Canvas revisado en `references/ML_Canvas_MiBici.pdf`.
- Ejecutar el notebook de exploración con el entorno del proyecto.
- Completar el README en el PR de documentación final con comandos verificados,
  cifras reales y resultados. Hay un README de plantilla dentro del proyecto que
  debe reemplazarse o enlazarse claramente con el principal.
- Usar `references/arquitectura.svg` como figura; el diagrama Mermaid editable y
  su justificación están en `arquitectura.md`. Revisar su presentación al exportar el PDF.
- Preparar `TC5061_Propuesta_Proyecto_Equipo27.pdf` con Canvas, arquitectura,
  decisiones, conteos, resultados, las cuatro capturas, estructura y referencias.
- Verificar acceso del profesor, protección de main y participación de los cuatro
  integrantes. Estas configuraciones de GitHub no se deducen del código local.

El PR de `preprocessingBranch` permanece independiente. Esta implementación no
requiere sus archivos para descargar o preparar datos.
