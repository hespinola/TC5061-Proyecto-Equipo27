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
make test
make data
make data
make features
make baseline
make tracking
```

La segunda ingesta verifica la versión fijada. El comando de baselines requiere
código confirmado en Git y registra cuatro runs en `mibici-equipo27`. La UI local
se abre en [http://127.0.0.1:5000](http://127.0.0.1:5000).

- [Manifiesto](references/data_manifest.json).
- [Arquitectura y propuesta](references/arquitectura.md).
- [Resultados e interpretación](reports/resultados_base.md).
- [Guía de ejecución y evidencias](references/ejecucion-y-evidencias.md).
- [Exploración](notebooks/1.0-equipo27-exploracion-mibici.ipynb).

Los datos, modelos, credenciales y archivos locales de MLflow se excluyen de Git.
La ejecución/conclusiones del notebook, el Canvas dentro del repositorio y el PDF
académico siguen pendientes de incorporar; consultar la lista de cierre principal.
