# feat: Model Registry, selección y reporte de experimentos

Base: `feat/mlflow-experiments` (parte B). Rama: `feat/mlflow-registry`.
Actualizar el destino a `main` después de integrar las partes A y B.

La selección usa validación con reglas explícitas de estabilidad y costo; prueba
solo se reporta. El nuevo punto de entrada `register` valida las corridas y el
linaje, registra tres modelos, asigna `champion` y verifica carga y paridad de
predicción por nombre/alias. P2 conserva su umbral de validación en el estimador.

Los elegidos fueron HGB 160/8/0.08 para P1/P2 y PCA8 + K-Means k=4 para P3.
El reporte documenta la caída de AP de P2 en prueba y el silhouette negativo de P3
en validación; el alias representa la producción académica local, sin afirmar
un despliegue de inferencia real ni beneficio operativo demostrado.

Incluye reporte PDF, fuente editable Markdown, capturas reales de MLflow,
resultados de registro e instrucciones de consumo y reproducción en README.
Los datos, base de tracking y modelos permanecen fuera del repositorio.

Validación: 50 tests locales; suite B completa (4 baselines + 9 candidatas);
`mlflow run . -e register --env-manager local` terminó y verificó tres cargas
por alias con paridad de predicciones. El reporte PDF fue renderizado e
inspeccionado visualmente. Se conservaron los intentos fallidos de verificación
como evidencia de depuración, sin usarlos para seleccionar los modelos.

Revisión solicitada: selección sin uso de prueba, linaje y aliases, limitaciones
de P2/P3, comandos y reporte. No fusionar sin revisión de otra persona.

Estado: texto preparado localmente; no representa un PR ya publicado ni revisado.
