# Preprocesamiento de datos — MiBici

## Descripción

Este módulo permite consolidar y preparar los datos históricos de viajes de MiBici para su posterior análisis y utilización en modelos de Machine Learning.

## Funcionalidades

- Identifica automáticamente los archivos CSV de viajes disponibles en el directorio de datos.
- Carga los archivos utilizando la codificación correspondiente.
- Concatena los registros de distintos periodos en un único DataFrame.
- Integra la información de nomenclatura de estaciones mediante sus identificadores.
- Agrega nombres, coordenadas geográficas y capacidades de las estaciones de origen y destino.
- Exporta un archivo CSV consolidado utilizando UTF-8.

## Prerequisitos

- Todos los archivos a concatenar deben encontrarse dentro de la carpeta data
- El archivo de nomenclatura también debe encontrarse dentro de la carpeta data 

**Nota: El script puede procesar n número de archivos siempre y cuando se encuentren presentes en la carpeta data**

## Archivos de entrada

- `datos_abiertos_YYYY_MM.csv`: registros históricos de viajes.
- `nomenclatura_YYYY_MM.csv`: información descriptiva y geográfica de estaciones.

## Resultado

El procesamiento genera el archivo `MiBici_datos_base.csv`, que contiene los registros originales enriquecidos con información de las estaciones de origen y destino.

El archivo resultante servirá como fuente de datos para los modelos de regresión, clasificación y clustering del proyecto.

## Consideraciones

- Los archivos de viajes deben mantener una estructura compatible.
- El procesamiento se ejecuta localmente y no requiere conexiones a servicios externos.