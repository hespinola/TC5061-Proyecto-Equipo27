# Model Card — MiBici P1: Demanda horaria

**Equipo 27 · TC5061 · 27 de noviembre de 2026**

## Resumen del modelo

| Campo | Descripción |
| --- | --- |
| Modelo registrado | `mibici-equipo27-p1-demanda` |
| Versión / alias original | v1 / `champion` |
| Algoritmo | HistGradientBoostingRegressor |
| Objetivo | Estimar salidas de bicicletas por estación y hora |
| MAE / RMSE de prueba | 0.7974 / 1.4696 salidas |
| Limitaciones | Estaciones nuevas, cambios de demanda y ausencia de inventario en tiempo real |
| Monitoreo | 9/16 variables con drift; comparación offline recomienda conservar A |

## 1. Identificación y linaje

- Responsable del proyecto académico: Equipo 27.
- Registry original documentado el 26 de octubre de 2026: `models:/mibici-equipo27-p1-demanda@champion`.
- Run ID original: `683b78f7fe9643be9b7496a77c951e58`.
- Artefacto original: `models:/m-e1de7c54c2b34622b72882cf2ba33c65`.
- Commit fuente: `9cd40eb0af9feed8bcbf6048a74292e04b8709c2`.
- Versión de datos: `ea4e65629d4c5f0565956bdb37c17920b8529a7a1308fd6ba7ecb74429f231d1`.
- Hiperparámetros documentados: 160 iteraciones (`n_estimators` en el reporte), profundidad máxima 8, tasa de aprendizaje 0.08.

La carga por alias y la paridad de predicción en cinco ejemplos se verificaron en el Registry original. Esta ficha no certifica que una instancia de MLflow distinta sirva exactamente ese artefacto: las reconstrucciones locales tienen otros identificadores.

## 2. Uso previsto y usos no previstos

El modelo estima el número de viajes que comienzan en una estación de MiBici del Área Metropolitana de Guadalajara durante una hora, con información disponible al inicio de la hora. Puede apoyar la planeación operativa y el análisis de demanda.

No estima directamente inventario, disponibilidad ni ocupación en tiempo real. No debe utilizarse como única fuente para decisiones automáticas de rebalanceo o asignación de recursos. No se han demostrado beneficios operativos mediante un experimento en producción.

## 3. Datos y características

- Fuente: registros históricos de MiBici, octubre de 2025 a septiembre de 2026.
- Dataset general: 4,497,119 viajes válidos y 4,125,960 filas estación-hora.
- Entrenamiento elegible de P1: 3,006,864 filas; del 8 de octubre de 2025 al 30 de junio de 2026 tras el calentamiento de los rezagos.
- Validación: julio de 2026 (350,424 filas); prueba: agosto-septiembre de 2026 (689,544 filas).
- Cohorte original de 471 estaciones.
- 16 características: hora, día de semana, fin de semana, mes, codificación cíclica de hora y día, rezagos y medias históricas de salidas y llegadas en ventanas de 24 y 168 horas.

Se utiliza una partición temporal y características basadas en observaciones previas para evitar fuga de información. El tratamiento de horas sin viajes como cero no permite determinar si la estación estuvo operativa.

## 4. Evaluación del modelo original

| Conjunto | Filas | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Validación | 350,424 | 0.7192 | 1.3445 |
| Prueba | 689,544 | 0.7974 | 1.4696 |
| Baseline 168 h (prueba) | 689,544 | 0.9337 | 1.9212 |

La reducción relativa del MAE de prueba frente al baseline semanal es aproximadamente 14.6%. El MAE y el RMSE se expresan en número de salidas. La selección se realizó con MAE de validación y, para candidatos cercanos, menor dispersión semanal y tamaño; el conjunto de prueba no se utilizó para elegir el modelo.

## 5. Limitaciones y uso responsable

- La demanda observada no equivale a demanda insatisfecha: una estación vacía puede impedir viajes.
- Las estaciones nuevas, eventos especiales y cambios en el patrón de movilidad pueden reducir la calidad de las predicciones.
- El monitoreo de octubre conserva la cohorte original; dos estaciones nuevas se reportaron fuera de ella y tres estaciones de la cohorte no registraron viajes durante el mes.
- Las métricas agregadas pueden ocultar diferencias de error entre estaciones y horarios. Se recomienda evaluar esos segmentos antes de usar el modelo para decisiones de distribución.
- Limitar las entradas a variables necesarias para la inferencia y no incorporar identificadores personales de viajeros. Mantener revisión humana y trazabilidad de decisiones.

## 6. Monitoreo de drift

El análisis retrospectivo con Evidently 0.7.23 compara el entrenamiento (3,006,864 filas elegibles) con octubre de 2026 (350,424 filas elegibles). Se utilizaron muestras de 100,000 filas de cada conjunto, semilla 42, Wasserstein para variables numéricas y Jensen-Shannon para categóricas.

Con umbral de distancia 0.1 y proporción de alerta 50%, nueve de las 16 entradas presentan drift (56.25%): mes y ocho variables históricas de salidas y llegadas. El control octubre 2025 frente a octubre 2026 identifica cambios en ocho de 16 entradas. No atribuye por sí solo una causa estacional.

La alerta justifica evaluar el error con datos recientes; no demuestra automáticamente pérdida de desempeño ni activa por sí sola una promoción.

## 7. Comparación de versiones y decisión

Comparación offline sobre 192,168 filas comunes del 15 al 31 de octubre de 2026. A es una reconstrucción del modelo original, no una afirmación de identidad binaria. B incorpora datos hasta el 14 de octubre con los mismos hiperparámetros.

| Variante local | Filas de entrenamiento | MAE | RMSE |
| --- | ---: | ---: | ---: |
| A — v1 | 3,006,864 | 0.998284 | 1.752160 |
| B — v2 | 4,205,088 | 0.998563 | 1.752562 |

Criterio: promover B solo si reduce el MAE al menos 1% y no aumenta el RMSE. B no cumple; se recomienda conservar A. El ensayo no cambia automáticamente el alias `champion` y no asigna tráfico real. Un holdout de 17 días no demuestra superioridad permanente; para B, el periodo de prueba histórico ya forma parte de su entrenamiento.

Run de comparación: `154915f09f7540838068073e3616e8e8`. Runs locales A y B: `7550a13afb87449d9405517af5865da1` y `95531ac048284cb494920fdde7b5ac9d`.

## 8. Operación y mantenimiento recomendado

El servicio de inferencia expone `POST /predict/demanda`. Verificar el modelo realmente servido mediante los metadatos de la respuesta y `/health`. Distinguir el modelo real de los artefactos sintéticos utilizados por CI/CD.

Como práctica de mantenimiento recomendada: revisar calidad de datos y drift periódicamente, evaluar MAE y RMSE con observaciones recientes y por segmentos, documentar resultados y solicitar revisión del equipo antes de cambiar el alias. No se afirma que exista reentrenamiento continuo automático en producción.

## 9. Evidencias versionadas

- [Resultados de experimentación](../reports/experiment_results.json)
- [Registro y selección de modelos](../reports/registry_results.json)
- [Resumen de drift](../reports/drift/resumen_p1_2026-10.json)
- [Resultados A/B](../reports/ab_testing/resultados_p1.json)
- [README general](../../README.md)
