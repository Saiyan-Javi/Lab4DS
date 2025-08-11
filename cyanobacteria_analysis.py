"""Herramientas para descargar y analizar floraciones de cianobacterias
usando Sentinel‑2 y la API de openEO.

El módulo implementa los siguientes pasos:

1. Conexión al backend de Copernicus mediante openEO.
2. Descarga de imágenes en formato GeoTIFF para una geometría dada.
3. Cálculo del índice de cianobacterias (NDCI) y de los índices NDVI y NDWI.
4. Conversión de los resultados a arreglos de ``numpy``.
5. Análisis temporal (media diaria y detección de picos) y espacial
   (mapas estáticos e interactivos) por lago.
6. Correlación entre NDCI, NDVI y NDWI.

El código está pensado como guía para completar el laboratorio propuesto.
Se asume que el usuario dispone de archivos GeoJSON con las geometrías de
los lagos de interés y que tiene acceso a un usuario de Copernicus Data
Space para autenticarse con openEO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import openeo
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

try:  # folium is opcional; permite mapas interactivos
    import folium
except Exception:  # pragma: no cover - folium no es obligatorio
    folium = None

# ---------------------------------------------------------------------------
# Configuración y conexión
# ---------------------------------------------------------------------------

BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL2_L2A"
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]


def connect_backend() -> openeo.Connection:
    """Devuelve una conexión autenticada al backend de Copernicus."""
    return openeo.connect(BACKEND_URL).authenticate_oidc()


# ---------------------------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------------------------


def download_lake(
    connection: openeo.Connection,
    geometry: Dict,
    start: str,
    end: str,
    outfile: Path,
) -> Path:
    """Descarga un GeoTIFF multibanda para la geometría dada.

    Parameters
    ----------
    connection:
        Conexión autenticada de openEO.
    geometry:
        Geometría en formato GeoJSON (feature["geometry"]).
    start, end:
        Fechas ISO (``YYYY-MM-DD``) que cubren al menos seis meses.
    outfile:
        Ruta de salida donde se guardará el archivo ``.tif``.
    """

    datacube = connection.load_collection(
        COLLECTION,
        spatial_extent=geometry,
        temporal_extent=[start, end],
        bands=BANDS,
    )

    job = connection.create_job(datacube.save_result(format="GTIFF"))
    job.start_and_wait().download_results(str(outfile))
    return outfile


# ---------------------------------------------------------------------------
# Cálculo de índices
# ---------------------------------------------------------------------------


def _read_raster(path: Path) -> Tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        transform = src.transform
    return data, transform


def cyano_index(b04: np.ndarray, b05: np.ndarray) -> np.ndarray:
    """Calcula el índice NDCI (aprox. Script CyanoDetection).

    NDCI = (B05 - B04) / (B05 + B04)
    """

    ndci = (b05 - b04) / (b05 + b04)
    return ndci


def ndvi(b04: np.ndarray, b08: np.ndarray) -> np.ndarray:
    return (b08 - b04) / (b08 + b04)


def ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    return (b03 - b08) / (b03 + b08)


@dataclass
class Indices:
    ndci: np.ndarray
    ndvi: np.ndarray
    ndwi: np.ndarray


def compute_indices(tif_path: Path) -> Indices:
    """Carga un GeoTIFF y calcula NDCI, NDVI y NDWI."""

    data, _ = _read_raster(tif_path)
    # Orden de bandas definido en BANDS
    b02, b03, b04, b05, b08, b11 = data
    return Indices(
        ndci=cyano_index(b04, b05),
        ndvi=ndvi(b04, b08),
        ndwi=ndwi(b03, b08),
    )


# ---------------------------------------------------------------------------
# Análisis temporal
# ---------------------------------------------------------------------------


def temporal_stats(indices: Iterable[Indices]) -> pd.DataFrame:
    """Genera una tabla con la media de los índices por fecha."""

    records = []
    for date_idx, ind in enumerate(indices):
        records.append(
            {
                "fecha": date_idx,
                "ndci_media": float(np.nanmean(ind.ndci)),
                "ndvi_media": float(np.nanmean(ind.ndvi)),
                "ndwi_media": float(np.nanmean(ind.ndwi)),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_temporal(df: pd.DataFrame, lake_name: str) -> None:
    """Grafica la evolución temporal del índice de cianobacterias."""

    plt.figure(figsize=(8, 3))
    plt.plot(df["fecha"], df["ndci_media"], marker="o")
    plt.title(f"Evolución temporal NDCI - {lake_name}")
    plt.xlabel("Fecha (índice)")
    plt.ylabel("NDCI medio")
    plt.grid(True)

    peaks, _ = find_peaks(df["ndci_media"], prominence=0.05)
    plt.scatter(df["fecha"].iloc[peaks], df["ndci_media"].iloc[peaks], color="red")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Análisis espacial
# ---------------------------------------------------------------------------


def plot_spatial(index: np.ndarray, transform: rasterio.Affine, title: str) -> None:
    """Muestra un mapa simple de la distribución espacial del índice."""

    plt.figure(figsize=(5, 5))
    plt.imshow(index, cmap="viridis")
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")


def map_interactive(index: np.ndarray, transform: rasterio.Affine, title: str):
    """Genera un mapa interactivo usando Folium.

    Requiere el paquete `folium` y que `index` esté georreferenciado.
    """

    if folium is None:
        raise RuntimeError("folium no está disponible")

    height, width = index.shape
    bounds = rasterio.transform.array_bounds(height, width, transform)
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2])
    folium.raster_layers.ImageOverlay(
        image=index,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        colormap=lambda x: (0, 1 - x, x, 1),
        opacity=0.6,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# Correlación de índices
# ---------------------------------------------------------------------------


def correlate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la correlación entre NDCI, NDVI y NDWI."""
    return df[["ndci_media", "ndvi_media", "ndwi_media"]].corr()


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse, json

    parser = argparse.ArgumentParser(description="Analiza cianobacterias en lagos")
    parser.add_argument("geojson", nargs=2, help="Rutas a archivos GeoJSON de los lagos")
    parser.add_argument("start", help="Fecha inicial (YYYY-MM-DD)")
    parser.add_argument("end", help="Fecha final (YYYY-MM-DD)")
    parser.add_argument("outdir", help="Directorio de salida")
    args = parser.parse_args()

    conn = connect_backend()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for lake_path in args.geojson:
        with open(lake_path) as f:
            geom = json.load(f)["features"][0]["geometry"]
        lake_name = Path(lake_path).stem
        tif = download_lake(conn, geom, args.start, args.end, outdir / f"{lake_name}.tif")
        ind = compute_indices(tif)
        df = temporal_stats([ind])
        plot_temporal(df, lake_name)
        plt.savefig(outdir / f"{lake_name}_temporal.png", dpi=150)
        corr = correlate(df)
        corr.to_csv(outdir / f"{lake_name}_corr.csv")
