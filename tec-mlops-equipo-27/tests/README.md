# Verificación del módulo de variables

Con las dependencias del proyecto instaladas, ejecutar desde
`tec-mlops-equipo-27/`:

```bash
python -m unittest discover -s tests -v
```

Las pruebas utilizan datos sintéticos y archivos temporales; no requieren viajes
descargados ni un servidor de MLflow. Incluyen una ejecución real del comando
`python -m mibici.features` con entradas y salida Parquet.

Se verifican los conteos del panel, horas sin viajes, estaciones sin nomenclatura,
capacidades inválidas, uniones sin multiplicación de registros, independencia
entre estaciones, rezagos, ventanas móviles, acumulado diario y límites temporales.

## Uso del corte temporal de P2

Construir primero las variables sobre el historial completo, para conservar el
contexto anterior a validación y prueba. Después filtrar cada partición:

```python
from mibici.config import FIN_ENTRENAMIENTO
from mibici.features import filtrar_periodo

train_p2 = filtrar_periodo(
    panel,
    fin=FIN_ENTRENAMIENTO,
    col_fin_objetivo="objetivo_fin_periodo",
)
```

Esto excluye las ventanas futuras que cruzan el corte y las que están incompletas.
Repetir para validación y prueba con sus fechas correspondientes. P1 puede filtrar
sin `col_fin_objetivo`, pues su objetivo solo cubre la hora de cada fila.

Al entrenar, seleccionar explícitamente los predictores: los conteos `salidas`,
`llegadas`, `flujo_neto` de la hora actual y las columnas `objetivo_` no deben entrar
en las variables de entrada de P1 ni de P2. Las filas sin capacidad válida carecen
de etiqueta de P2 y deberán reportarse y excluirse de su evaluación.
