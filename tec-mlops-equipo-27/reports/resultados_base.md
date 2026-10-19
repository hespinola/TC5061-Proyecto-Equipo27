# Resultados del experimento base · MiBici, equipo 27

Ejecución reportada por el equipo el 18 de octubre de 2026, a las 20:35 de Ciudad
de México. Los cuatro runs terminaron; el proceso completo reportó 14.70 segundos,
incluida la reconstrucción del panel a partir del Parquet ya preparado. Ese tiempo
no incluye descarga ni ingesta de los CSV.

- Commit ejecutado: `d921472702dc1c0b2a3af8ca3179b0567e913c8e`.
- Versión de datos: `ea4e65629d4c5f0565956bdb37c17920b8529a7a1308fd6ba7ecb74429f231d1`.
- Fuente de cifras: `reports/baseline_results.json` y figuras generadas por el código.
- Entrenamiento: octubre de 2025–junio de 2026; validación: julio; prueba: agosto–septiembre.
- Los resultados corresponden a simulación histórica y no demuestran impacto operativo.

## P1 · Demanda por estación y hora

| Referencia | MAE validación | RMSE validación | MAE prueba | RMSE prueba |
| --- | ---: | ---: | ---: | ---: |
| Misma hora de la semana anterior, 168 h | 0.868982 | 1.824453 | 0.933745 | 1.921200 |
| Misma hora del día anterior, 24 h | 0.910708 | 1.966678 | 1.035465 | 2.194621 |

Cada referencia se evalúa en 350,424 filas de validación y 689,544 de prueba,
con la misma población de estaciones y horas en esas particiones. La referencia
semanal tiene menor MAE y RMSE en validación y también en prueba. El MAE está
expresado en viajes por estación y hora; las figuras muestran totales diarios solo
como visualización, no como la granularidad de evaluación.

Runs:

- Semanal: `9ea871aaa9ce4ebaae7a04bdf7a25f82`.
- Diaria: `14a58f067827468db3516263d531c7f9`.

## P2 · Proxy de desequilibrio futuro

Baseline: frecuencia de positivos por estación y hora calculada con entrenamiento.
Umbral elegido en validación: **0.30**, mediante máximo F1 y desempates documentados.
La etiqueta indica magnitud del flujo neto total de tres horas ≥ 50 % de capacidad;
no observa estaciones vacías/llenas ni inventario real.

| Métrica | Validación | Prueba |
| --- | ---: | ---: |
| Precision | 0.403668 | 0.453065 |
| Recall | 0.314783 | 0.269737 |
| F1 | 0.353727 | 0.338152 |
| Average precision | 0.309132 | 0.325776 |
| Proporción de positivos | 0.016678 | 0.021816 |
| Filas evaluadas | 347,998 | 685,678 |

La matriz de prueba contiene 665,848 verdaderos negativos, 4,871 falsos positivos,
10,924 falsos negativos y 4,035 verdaderos positivos. Detecta aproximadamente el
27 % de las filas positivas y deja sin detectar alrededor del 73 %. Esta limitación
es relevante para una futura priorización de rebalanceo, aunque todavía no se mide
su costo operativo. La prevalencia de prueba es aproximadamente 2.18 %.

Se excluyen 17,520 filas sin capacidad válida (dos estaciones × 8,760 horas).
También se respetan el calentamiento semanal y las ventanas completas dentro de
cada partición. Las ventanas solapadas corresponden a predicciones horarias;
no deben interpretarse como incidentes operativos independientes.

Run: `843c94bd23b64dc88a732d3b027aa333`.

## P3 · Agrupamiento de perfiles de uso

- Estaciones agrupadas: **424**; excluidas por perfil no elegible: **47**.
- Grupos elegidos: **6**, el mejor silhouette dentro del rango ensayado de 2 a 6.
- Silhouette de entrenamiento: **0.083076**.
- Davies-Bouldin de entrenamiento: **1.953043**.

El silhouette cercano a cero sugiere separación débil entre los grupos en este
espacio de variables. No se presenta como una segmentación operativa consolidada,
ni como un óptimo global de número de grupos. La estabilidad entre semillas y
periodos sigue siendo un análisis posterior. Las coordenadas se utilizan para el
mapa, no para construir los grupos.

Run: `becb2a5062f840708a5f7b30fd0d77f9`.

## Evidencias y trabajo pendiente

Las figuras `p1_rezago_168.png`, `p1_rezago_24.png`, `p2_confusion.png` y
`p3_grupos.png` fueron generadas y se revisó su contenido. Las cuatro capturas
reales de la interfaz de MLflow están incorporadas y revisadas. El detalle,
artefacto y comparación de P1 muestran una ejecución previa con el mismo commit,
versión de datos y métricas; sus IDs difieren de la ejecución posterior identificada
en este reporte. La lista muestra las dos ejecuciones completas.

Faltan las salidas/conclusiones del notebook de exploración, la integración del
Canvas y el cierre del README y del PDF. La presencia de estas evidencias no implica
que toda la entrega académica esté terminada.
