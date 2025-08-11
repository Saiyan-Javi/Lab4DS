# Lab4DS
Laboratorio 4 de Data Science 2025

## Contenido

- `Lab 4.ipynb`: cuaderno original del laboratorio.
- `cyanobacteria_analysis.py`: script en Python que muestra cómo conectar a la
  API de [openEO](https://openeo.org/) para descargar imágenes Sentinel‑2 y
  realizar un análisis temporal y espacial de floraciones de cianobacterias.

## Uso básico del script

```bash
pip install openeo rasterio numpy pandas matplotlib scipy folium
python cyanobacteria_analysis.py Lago_Atitlan.geojson Lago_Amatitlan.geojson \
    2024-01-01 2024-07-01 resultados
```

El comando anterior descargará las bandas requeridas para cada lago, calculará
el índice de cianobacterias (NDCI), NDVI y NDWI, generará gráficos temporales y
almacenará los productos en el directorio `resultados/`.
