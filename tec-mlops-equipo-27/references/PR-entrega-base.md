# Ingesta reproducible, baselines y exploración de MiBici

## Problema y resultado esperado

El repositorio tenía ejemplos de plantilla en ingesta y entrenamiento. Este cambio
añade una ruta ejecutable desde los CSV oficiales hasta el panel y cuatro runs:
referencias semanal/diaria de P1, frecuencia histórica para P2 y K-Means para P3.
Se apoya en las correcciones de `feat/features`; integrar ese PR antes de este.

## Cambios

- Descarga por configuración, normalización, procesamiento por bloques, control
  de duplicados y Parquet con esquema fijo.
- Manifiesto de fuentes, hashes, conteos y versión; la reconstrucción rechaza cambios.
- Panel horario con cobertura explícita y llegadas posteriores al periodo reportadas.
- Baselines con cortes temporales, selección de umbral en validación y tracking.
- Notebook de exploración, propuesta de arquitectura y guía de evidencias.
- Comandos de datos, variables, baselines, tracking y pruebas.

## Validación

Validación reportada por el equipo:

- `pip check`: sin conflictos de dependencias.
- Suite inicial: 27 pruebas satisfactorias. Confirmar la suite ampliada tras el ajuste HTTP 406.
- Ingesta real: 12 meses, 4,497,877 registros originales y 4,497,119 viajes válidos.
- Reproducción: segunda ingesta con la misma versión y hashes.
- Panel: 4,125,960 filas y 471 estaciones; 16 llegadas posteriores al cierre.
- Experimentos en MLflow: pendientes de ejecución y revisión.

Versión de datos:
`ea4e65629d4c5f0565956bdb37c17920b8529a7a1308fd6ba7ecb74429f231d1`.

## Alcance posterior

La actualización final del README y el PDF requiere las cifras de ingesta, métricas,
capturas de la UI y los datos completos de integrantes. El PR de preprocessing de
Humberto permanece abierto y no es una dependencia de esta implementación.
