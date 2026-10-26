# feat: experimentación y tracking avanzado de P1, P2 y P3

Base: `feat/mlflow-project` (parte A de Dani). Destino posterior: `main` una vez
integrada la parte A. Rama propuesta: `feat/mlflow-experiments`.

El Project ahora ejecuta cuatro baselines y nueve candidatas desde el punto de
entrada `experiment`, con una búsqueda padre por pregunta y tres hijas por
combinación. Cada modelo conserva firma, ejemplo, manifiesto, inputs agregados,
configuración y diagnósticos. P2 sirve el umbral de validación; P3 registra la
estabilidad entre semillas y periodos en una geometría comparable.

Se preservan datos, cortes temporales y métricas anteriores; la ejecución completa
verificó paridad de los cuatro baselines contra sus IDs y valores originales.
`reports/experiment_results.json` conserva los IDs reales, parámetros y métricas.
No se incluyen datos, SQLite, artefactos de MLflow ni estimadores serializados.

Validación: 45 tests locales y ejecución real de
`mlflow run . -e experiment --env-manager local` en el servidor SQLite del equipo.
Se corrigió la serialización explícita de estimadores propios para MLflow 3.17.
El intento previo fallido se conserva en tracking y no entra en la comparación.

Revisión solicitada: confirmar las variables anteriores a t, la purga de P2 y la
comparabilidad de P3. El reporte/Registry de la parte C se revisa en otro PR.

Estado: texto preparado localmente; no representa un PR ya publicado ni revisado.
