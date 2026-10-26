# MiBici · módulo del equipo 27

Base reproducible de Machine Learning para viajes de MiBici. El proyecto generado
con Cookiecutter Data Science se encuentra en esta carpeta; el repositorio Git
está un nivel arriba.

**La documentación principal está en el [README del repositorio](../README.md)**:
propósito, preguntas, integrantes y roles, datos, particiones, instalación,
comandos, arquitectura, baselines, resultados, MLflow y colaboración.

Con Python 3.13 y el entorno del proyecto activado, ejecutar desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

Para reproducir las partes B y C, volver a la raíz Git con `cd ..` y seguir
la [guía de MLflow Project](../README.md#mlflow-project-y-servidor-de-tracking):
iniciar el servidor de tracking, definir las variables de entorno y ejecutar
`ingest`, `experiment` y `register` mediante `mlflow run`, en ese orden.
La guía incluye instalación y variables para Bash y Windows PowerShell.

La experimentación repite cuatro baselines y evalúa nueve candidatas en
`mibici-equipo27`. El registro selecciona un modelo por pregunta y verifica su
carga con el alias `champion`. El entrenamiento requiere código confirmado en
Git. La UI local se abre en [http://127.0.0.1:5000](http://127.0.0.1:5000).

- [Manifiesto](references/data_manifest.json).
- [Arquitectura y propuesta](references/arquitectura.md).
- [Resultados e interpretación](reports/resultados_base.md).
- [Guía de ejecución y evidencias](references/ejecucion-y-evidencias.md).
- [Exploración](notebooks/1.0-equipo27-exploracion-mibici.ipynb).
- [Reporte de experimentación y Registry](reports/reporte_mlflow.md).
- [Resultados de experimentación](reports/experiment_results.json).
- [Modelos registrados y verificación por alias](reports/registry_results.json).
- [Reporte PDF de partes B y C](../output/pdf/TC5061_Reporte_MLFlow_Equipo27.pdf).

Los datos, modelos, credenciales y archivos locales de MLflow se excluyen de Git.
Al clonar, hay que reproducir la ingesta, experimentación y registro para disponer
de los modelos en un servidor propio. Los JSON y las capturas son evidencia de
la ejecución documentada; no incluyen la base de tracking ni los modelos.
