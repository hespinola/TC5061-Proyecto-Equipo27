# Validación local de partes B y C

Ejecutada el 25 de octubre de 2026 (America/Mexico_City).

- Python 3.13.9; dependencias principales fijas ya instaladas, MLflow 3.17.1.
- 50 tests terminaron correctamente: `python -m unittest discover -s tests -q`
  desde `tec-mlops-equipo-27/`.
- `ruff check` de los módulos modificados y `ruff format --check` sin errores.
  La revisión global detecta una regla heredada TRY004 en `dataset.py:107`
  (ValueError en la validación de zona horaria), fuera de estos cambios.
- La ingesta previa se reutilizó con validación SHA-256 del manifiesto y Parquet.
  La suite reconstruyó el panel completo y reprodujo las métricas de los cuatro
  baselines históricos antes de aceptar el resumen final.
- `mlflow run . -e experiment --env-manager local`: terminó; corrida
  `580a6efc0ebd4aea97160d1a40dc1e75`, commit
  `9cd40eb0af9feed8bcbf6048a74292e04b8709c2`.
- `mlflow run . -e register --env-manager local`: terminó; corrida
  `39e56e8397a744138f03d57715ce5d45`, commit
  `24e7852b25a16292b4967009a60c515a0ff43ee2`.
- Tres versiones 1 con `champion`: firma y ejemplo presentes, carga por versión
  y alias, y paridad de predicciones comprobadas. Los detalles están en
  `reports/registry_results.json`.
- PDF con tablas provenientes de los JSON, ocho capturas reales de la UI,
  linaje y referencias APA 7; se revisó el render de las 16 páginas.

No se publicaron ramas ni PR porque la solicitud de publicación fue rechazada.
La evidencia de colaboración de cuatro integrantes sigue pendiente. La ejecución
no volvió a instalar el entorno desde cero ni volvió a descargar datos de MiBici.
Los comandos para hacerlo se documentan sin presentar esos pasos como ejecutados.

El servidor activo usa `equipo27/mlflow.db` y `equipo27/mlartifacts/` de la copia
local de esta tarea, ambos excluidos de Git. Tras clonar en otra máquina se
reconstruyen con los puntos de entrada del Project. No usar como base otra
SQLite vacía si se desea consultar las capturas y los IDs de esta ejecución.
