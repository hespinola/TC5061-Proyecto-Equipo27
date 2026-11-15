# Monitoreo de P1 para el Entregable 8

Este procedimiento vigila las 16 entradas del modelo de demanda por estación y
hora. El notebook de Oscar conserva el análisis de viajes como complemento e
incorpora una sección para ejecutar este monitoreo principal.

## Conjuntos y trazabilidad

- Referencia: filas elegibles de octubre de 2025 a junio de 2026. Se reconstruye
  con la preparación original y se retiran las filas sin todos los rezagos,
  igual que para el HGB. El conteo debe coincidir con el del run documentado.
- Validación original: julio de 2026. Prueba original: agosto y septiembre.
  Sus filas no se agregan a la referencia del reporte principal.
- Actual: octubre de 2026, posterior a todas las particiones originales.
- Contexto: últimas 168 horas de septiembre, utilizadas solamente para calcular
  los predictores disponibles al inicio de octubre. No se usan objetivos futuros.
- Control complementario: octubre de 2025 dentro de la referencia elegible. No
  incluye la primera semana sin rezagos. Comparar un solo mes de otro año no
  permite atribuir causalmente un cambio a la estacionalidad.

La versión base se identifica con los archivos de evidencia del Registry y de
experimentación. No se presupone que el alias champion siga apuntando a ella.
Se verifican data_version y el número de filas de entrenamiento. El resumen
incluye identidad del modelo, hash del CSV actual, hash del manifiesto, versiones
de paquetes y hashes del código utilizado.

Se simula la predicción al inicio de cada hora, con las observaciones anteriores
a esa hora. Los rezagos se actualizan conforme avanza el mes, igual que en la
evaluación histórica original.

El monitoreo reconstruye el panel histórico con guardar=False. Los Parquet
congelados, su manifiesto y los modelos anteriores permanecen como entradas.
La ingesta original conserva la escritura del panel por defecto.

## Población y calidad

El reporte principal mantiene la cohorte de estaciones del panel original.
Incluye todas sus horas, aun sin viajes observados, con el mismo convenio de
ceros del pipeline original. Un cero de viajes observados no prueba que la
estación estuviera operativa ni que la captura estuviera completa.

Las estaciones nuevas y los viajes fuera de la cohorte se contabilizan en el
resumen; no se les inventa historia de rezagos. El reporte de viajes complementa
el monitoreo al incluir la mezcla completa de estaciones del mes actual.

Las llegadas de viajes iniciados en septiembre y terminados en octubre se
incorporan al panel actual. Las llegadas posteriores al cierre de octubre se
excluyen de las horas de octubre. Se verifican los totales de salidas y llegadas
de la cohorte y la continuidad del contexto.

Se requiere al menos un viaje registrado por día del mes antes de completar la
malla con ceros. Esta comprobación detecta huecos diarios evidentes; no demuestra
por sí sola que cada hora tenga captura completa.

Los campos personales se excluyen al seleccionar las columnas del CSV actual.
Los duplicados idénticos se contabilizan y se conservan una vez; los IDs con
datos contradictorios provocan un error. Las coincidencias entre variables de
viajes se llaman combinaciones repetidas, no viajes duplicados.

## Ejecución

Desde el subdirectorio del proyecto, con su entorno Python activado:

    python -m pip install -r requirements.txt
    python -m pip check
    python -m unittest discover -s tests -v
    python -m mibici.monitoreo_p1
    make security-code security-secrets

El workflow Drift checks ejecuta las pruebas funcionales del monitoreo en los
PR que modifican este código. Incluye una comparación pequeña con Evidently
real para verificar que los veredictos exportados se interpretan correctamente.
No descarga datos de MiBici ni necesita un servidor MLflow.

La ingesta congelada debe existir y coincidir con el manifiesto. El CSV de octubre
se descarga con el cliente existente y se guarda en data/raw. El mes actual debe
ser contiguo al periodo congelado; otros meses requieren proporcionar antes su
contexto histórico.

La muestra por conjunto es reproducible, de hasta 100,000 filas, con la semilla del
proyecto. Se usan explícitamente Wasserstein normalizada para entradas numéricas
y Jensen-Shannon para calendario categórico. El umbral 0.1 y la proporción 0.5 son
parámetros exploratorios, no umbrales calibrados con un costo operativo.

Las categorías del calendario se convierten a etiquetas de texto únicamente en
la copia entregada a Evidently. Esto evita la incompatibilidad de sus métricas
de calidad con etiquetas numpy.int8. Las entradas originales del modelo
conservan sus tipos numéricos; una prueba de regresión verifica ese contrato.

Los reportes se generan en reports/drift:

- calidad_p1_2026-10_vs_entrenamiento.html
- drift_p1_2026-10_vs_entrenamiento.html
- drift_p1_2026-10_vs_2025-10.html
- Las correspondientes tablas CSV por variable.
- drift_p1_2026-10.png, con las distancias calculadas por Evidently.
- resumen_p1_2026-10.json, con resultados, calidad y trazabilidad.

Los HTML y CSV son artefactos locales regenerables según las reglas del
repositorio. El JSON y la gráfica pueden conservarse como evidencia de la
ejecución real. El notebook se guarda con las salidas limpias antes de subirlo.

## Interpretación y límite de esta entrega parcial

El cambio de distribución de mes es esperable por el diseño temporal. Las
codificaciones cíclicas del calendario tampoco constituyen evidencia
independiente entre sí. Deben examinarse junto con rezagos y promedios móviles.

Detectar drift no demuestra pérdida de desempeño ni justifica por sí solo
reentrenar. Este monitoreo no entrena una segunda versión, no hace una asignación
aleatoria de usuarios y no decide una promoción. La comparación de versiones
registradas y la recomendación de Continuous Training se desarrollarán en los
siguientes pasos, con un conjunto de evaluación separado del ajuste.

## Evidencia de validación

Validación local realizada el 14 de noviembre de 2026, con Evidently 0.7.23,
pandas 3.0.6, NumPy 2.5.4, PyArrow 25.0.1 y scikit-learn 1.9.1:

- 69 pruebas del proyecto ejecutadas correctamente, incluida la integración con
  Evidently y la regresión para calendario int8.
- Bandit sin hallazgos nuevos.
- CSV actual: 635,532 viajes válidos, con registros en los 31 días de octubre.
- Referencia elegible: 3,006,864 filas, exactamente el conteo del run documentado.
  Se excluyen 79,128 filas iniciales sin rezagos completos; la primera fila
  elegible es del 8 de octubre de 2025.
- Actual: 350,424 filas, correspondientes a 471 estaciones y 744 horas.
  Ninguna fila actual fue excluida por predictores faltantes.
- Se identificaron las estaciones 124 y 492 fuera de la cohorte original y
  176 salidas fuera de esa cohorte. Las estaciones 175, 405 y 407 no aparecen
  en los viajes del mes.
- Se incorporaron 16 llegadas de viajes históricos que cruzan a octubre.
- Se generaron los tres HTML, las dos tablas CSV, la gráfica PNG y el resumen
  JSON. Las tablas exportadas y el JSON tienen los mismos 16 veredictos.
- El manifiesto y los Parquet congelados superaron su verificación de hashes
  al terminar la ejecución.

Se detecta drift en 9 de las 16 entradas frente al entrenamiento: mes y las
ocho variables de rezagos y medias de salidas/llegadas. Frente al control de
octubre de 2025 persiste en las ocho variables de demanda. Esto requiere
evaluar el desempeño del modelo; todavía no determina una decisión de reentrenamiento.

La revisión de secretos identificó 10 valores del resumen que corresponden a
identificadores Git y hashes de trazabilidad. Se comprobaron contra la evidencia
del Registry, el manifiesto, el CSV y los archivos fuente y se registraron como
falsos positivos concretos en la baseline existente.

La evidencia completa se conserva en reports/drift/resumen_p1_2026-10.json.
Esta validación es local; el estado de los checks del commit que se va a fusionar
debe comprobarse en el PR.
