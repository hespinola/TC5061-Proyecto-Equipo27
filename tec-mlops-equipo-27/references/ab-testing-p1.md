# P1: versión alternativa y comparación A/B retrospectiva

## Diseño fijado antes de comparar

Se reconstruye localmente el HGB documentado de P1 porque no se dispone del
artefacto del registry original. Esta reconstrucción A no se presenta como el
mismo objeto binario ni como la versión vigente del servidor del equipo.
Ambos modelos se registran bajo `mibici-equipo27-p1-demanda` en un registry de
trabajo independiente, con versiones distintas y sus corridas de origen.

| Brazo | Entrenamiento | Evaluación común |
|---|---|---|
| A: reconstrucción | Filas elegibles de octubre 2025 a junio 2026 | Octubre 15–31 de 2026 |
| B: reentrenamiento | Histórico hasta septiembre 2026 más octubre 1–14 | Octubre 15–31 de 2026 |

Los dos utilizan las 16 entradas `FEATURES_P1`, semilla 42, 160 iteraciones,
profundidad 8, learning rate 0.08 y `early_stopping=False`. Estos valores se
leen de la corrida seleccionada documentada; no se buscan hiperparámetros en
el holdout. B cambia la ventana de entrenamiento, no la familia del modelo.
La validación y prueba históricas pasan a formar parte del entrenamiento de B;
sus métricas antiguas no sirven como prueba independiente de B.

La cohorte es la misma que en el monitoreo. Las últimas 168 horas de septiembre
permiten construir octubre; se incorporan las llegadas de viajes que cruzan de
septiembre a octubre. En cada hora de evaluación se usan observaciones anteriores
a esa hora, incluidas las de días previos del propio holdout. Esto reproduce la
inferencia secuencial; no es una predicción de todo el mes desde el 1 de octubre.
Las funciones de variables existentes impiden incluir el objetivo contemporáneo.

## Métricas y decisión

Se cargan por **número de versión** los dos modelos desde MLflow y se evalúan
sobre exactamente las mismas filas, verificables por su SHA-256. Se aplica a
ambos el recorte a cero y redondeo a cuatro decimales de la API.

- Métrica primaria: MAE de viajes por estación y hora.
- Salvaguarda: RMSE, para no aceptar mayor penalización de errores grandes.
- Diagnóstico: MAE y RMSE diarios sobre los 17 días del holdout.
- Regla previa: recomendar B si reduce el MAE al menos 1% y no aumenta RMSE.
  En otro caso, conservar A como referencia de esta comparación.

El 1% es un criterio práctico explícito, no una prueba de significancia ni un
umbral calibrado con costos operativos. La comparación offline satisface el
ejercicio de evaluar ambas versiones sobre el periodo actual; no es un experimento
aleatorizado con usuarios. Un único mes no demuestra superioridad permanente.
La recomendación se registra; **no se modifica el alias `champion`**.

## Resultado de la ejecución local

La ejecución del 15 de noviembre de 2026 (hora de México) registró las versiones
1 y 2 del mismo modelo y comparó 192,168 filas: 471 estaciones durante las
408 horas del 15 al 31 de octubre. La evidencia completa está en
[`resultados_p1.json`](../reports/ab_testing/resultados_p1.json).

| Modelo local | Filas de entrenamiento | MAE | RMSE |
|---|---:|---:|---:|
| A, versión 1 reconstruida | 3,006,864 | 0.998284 | 1.752160 |
| B, versión 2 reentrenada | 4,205,088 | 0.998563 | 1.752562 |

B aumenta el MAE aproximadamente 0.028% y también aumenta el RMSE. Se recomienda
conservar A como referencia; B no cumple el criterio previo de promoción.
La diferencia es pequeña y no se afirma significancia estadística. Este
reentrenamiento no aporta evidencia para sustituir el modelo: conviene mantener
el monitoreo y evaluar futuros candidatos en un periodo nuevo, sin reutilizar
este holdout para ajustar hiperparámetros. No se modificaron aliases.

La ejecución real utilizó Python 3.14.5 y las versiones de bibliotecas registradas
en el JSON. El proyecto y CI tienen como objetivo Python 3.13; en este equipo,
Windows bloqueó una extensión de SQLAlchemy en el entorno 3.13. Por ello, estos
resultados no certifican una ejecución completa bajo 3.13. El registry y sus
artefactos permanecen locales; el JSON permite revisar los resultados sin
incorporar la base de datos ni los modelos binarios al repositorio.

## Reproducción

Desde `tec-mlops-equipo-27/`, con el entorno del proyecto (Python 3.13 y sus
dependencias) instalado:

```bash
mkdir -p .cache/ab_testing
python -m mibici.ab_testing --desde-csv --tracking-uri sqlite:///.cache/ab_testing/mlflow.db
python -m pytest tests/test_ab_testing.py -q
```

PowerShell, desde el mismo subdirectorio y con el entorno activado:

```powershell
New-Item -ItemType Directory -Force .cache/ab_testing | Out-Null
python -m mibici.ab_testing --desde-csv --tracking-uri sqlite:///.cache/ab_testing/mlflow.db
python -m pytest tests/test_ab_testing.py -q
```

Usar una base de trabajo nueva para identificar claramente las versiones locales.
`--desde-csv` descarga o reutiliza los doce CSV históricos y verifica sus hashes,
tamaños, codificaciones, conteos de limpieza y unicidad global contra el manifiesto.
Construye únicamente las variables de P1 con las funciones existentes. Una prueba
compara esas variables con las de `preparar` sobre los mismos viajes. No usa
capacidad, coordenadas ni nombres de estaciones, que no forman parte de P1.
El CSV actual también debe coincidir con el hash de la evidencia de drift.
No se cambia el manifiesto ni se sobrescriben los Parquet congelados durante el A/B.

La ruta CSV evita una dependencia innecesaria del catálogo para P1: su URL de
septiembre dejó de responder y, aunque la copia original del CSV coincide,
su Parquet regenerado en este equipo no reproduce el hash congelado. No se
aceptó ese Parquet ni se modificó el hash esperado. Si ya se dispone de los dos
Parquet originales verificados, omitir `--desde-csv` para usar `preparar(guardar=False)`.
Las comprobaciones de la ingesta completa permanecen intactas.

`reports/ab_testing/resultados_p1.json` reúne versiones, corridas, ventanas,
huellas de entrenamiento y evaluación, métricas, decisión, entorno y limitaciones.
El protocolo se registra en la corrida padre antes del entrenamiento. Cada hijo
guarda parámetros, input de entrenamiento, modelo con firma y métricas del holdout.
El digest abreviado de MLflow usa los primeros 32 caracteres; la huella SHA-256
completa se conserva en tags y evidencia. El registro se verifica recargando
cada versión y comprobando paridad de predicciones.

La ejecución se niega a sobrescribir resultados existentes. Para una repetición
independiente usar `--salida` con otra ruta y conservar el protocolo original;
no ajustar la regla al observar el holdout.

## Pruebas

Las pruebas cubren separación temporal, integridad de la cohorte y horas,
predictores no finitos, invariancia del entrenamiento al alterar el holdout,
decisión conservadora y registro real de dos versiones en SQLite temporal.
Las pruebas sintéticas verifican el software; no son evidencia de desempeño
de MiBici ni se mezclan con el resultado real.
