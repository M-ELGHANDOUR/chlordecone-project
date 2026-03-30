"""
spatial.py — Analyse spatiale et cartographie
=============================================
Cours mobilisé : Data Engineering — Cartographie Spatiale (09 Mars 2026)
Concepts : GeoPandas, shapefiles, projections EPSG, cartes choroplèthes,
           heatmaps de contamination, reprojection de coordonnées

Fonctions :
  - load_shapefile()       : chargement du shapefile Martinique
  - build_geodataframe()   : conversion du DataFrame en GeoDataFrame (EPSG:5490)
  - plot_spatial_map()     : carte de contamination spatiale (3 panels)
  - plot_choropleth()      : carte choroplèthe par commune
  - aggregate_by_commune() : agrégation spatiale des mesures par commune
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    logger.warning("⚠️  geopandas non installé. pip install geopandas")


# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SHP_DIR   = ROOT / "data" / "external"
REPORTS   = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

# Système de projection des coordonnées X/Y du dataset
# EPSG:5490 = RGAF09 / UTM zone 20N (Martinique / Guadeloupe)
EPSG_SOURCE = 5490
EPSG_WGS84  = 4326


def load_shapefile(name: str = "Martinique") -> "gpd.GeoDataFrame":
    """
    Charge un shapefile depuis data/external/.

    Parameters
    ----------
    name : str
        Nom de base du shapefile (sans extension), ex. "Martinique"

    Returns
    -------
    GeoDataFrame
    """
    if not HAS_GEO:
        raise ImportError("geopandas requis. pip install geopandas")

    shp_path = SHP_DIR / f"{name}.shp"
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Shapefile introuvable : {shp_path}\n"
            "Copiez les fichiers .shp/.dbf/.prj/.shx dans data/external/"
        )

    gdf = gpd.read_file(shp_path)
    logger.info(f"  📂 Shapefile chargé : {name} ({len(gdf)} entités, CRS: {gdf.crs})")
    return gdf


def build_geodataframe(df: pd.DataFrame) -> "gpd.GeoDataFrame":
    """
    Convertit le DataFrame pandas en GeoDataFrame avec les coordonnées X/Y.
    Système source : EPSG:5490 (UTM zone 20N — Martinique).

    Parameters
    ----------
    df : pd.DataFrame avec colonnes X et Y

    Returns
    -------
    GeoDataFrame en EPSG:5490
    """
    if not HAS_GEO:
        raise ImportError("geopandas requis.")

    df_valid = df.dropna(subset=["X", "Y"]).copy()

    geometry = [Point(x, y) for x, y in zip(df_valid["X"], df_valid["Y"])]
    gdf = gpd.GeoDataFrame(df_valid, geometry=geometry, crs=f"EPSG:{EPSG_SOURCE}")
    logger.info(f"  🗺️  GeoDataFrame créé : {len(gdf):,} points — CRS EPSG:{EPSG_SOURCE}")
    return gdf


def aggregate_by_commune(
    df: pd.DataFrame,
    gdf_communes: "gpd.GeoDataFrame",
    commune_col: str = "COMMU_LAB",
) -> "gpd.GeoDataFrame":
    """
    Agrège les statistiques de contamination par commune et les joint
    au GeoDataFrame des communes pour les cartes choroplèthes.

    Returns
    -------
    GeoDataFrame enrichi avec taux médian, % dépassement, N prélèvements.
    """
    agg = (
        df.groupby(commune_col)
        .agg(
            taux_median=("CHLORDECONE_RATE", "median"),
            taux_moyen=("CHLORDECONE_RATE", "mean"),
            pct_reglementaire=("ABOVE_REG_THRESHOLD", "mean"),
            n_prelevements=("CHLORDECONE_RATE", "count"),
        )
        .reset_index()
    )

    merged = gdf_communes.merge(
        agg, left_on="NOM_COM", right_on=commune_col, how="left"
    )
    logger.info(f"  ✅ Agrégation par commune : {agg.shape[0]} communes avec données")
    return merged


def plot_spatial_map(
    df: pd.DataFrame,
    gdf_basemap: "gpd.GeoDataFrame" = None,
    save: bool = True,
) -> plt.Figure:
    """
    Carte de contamination spatiale — 3 panels :
      A — Scatter coloré par log(CLD) sur fond de carte
      B — Heatmap de densité (hexbin)
      C — Classes de contamination (4 niveaux)
    """
    if not HAS_GEO:
        raise ImportError("geopandas requis.")

    gdf_pts = build_geodataframe(df)

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle(
        "Distribution Spatiale de la Contamination au Chlordécone (Martinique)",
        fontsize=15, fontweight="bold",
    )

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "pollution", ["#2D8A5E", "#F9A825", "#E8531A", "#C62828"], N=256
    )

    # Panel A — Taux log sur fond de carte
    ax = axes[0]
    if gdf_basemap is not None:
        gdf_basemap.plot(ax=ax, color="#e8e8e8", edgecolor="#aaaaaa", linewidth=0.5)
    sc = ax.scatter(
        gdf_pts["X"], gdf_pts["Y"],
        c=gdf_pts["LOG_CHLD"], cmap=cmap, s=3, alpha=0.6, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="log(1 + Taux CLD)")
    ax.set_title("A — Taux de contamination (log)")
    ax.set_axis_off()

    # Panel B — Hexbin densité
    ax = axes[1]
    if gdf_basemap is not None:
        gdf_basemap.plot(ax=ax, color="#e8e8e8", edgecolor="#aaaaaa", linewidth=0.5)
    hb = ax.hexbin(
        gdf_pts["X"], gdf_pts["Y"], gridsize=50,
        C=gdf_pts["CHLORDECONE_RATE"], reduce_C_function=np.median,
        cmap=cmap, alpha=0.8,
    )
    plt.colorbar(hb, ax=ax, label="Médiane CLD (mg/kg)")
    ax.set_title("B — Heatmap médiane par cellule hexagonale")
    ax.set_axis_off()

    # Panel C — Classes de contamination
    ax = axes[2]
    if gdf_basemap is not None:
        gdf_basemap.plot(ax=ax, color="#e8e8e8", edgecolor="#aaaaaa", linewidth=0.5)

    class_colors = {0: "#2D8A5E", 1: "#F9A825", 2: "#E8531A", 3: "#C62828"}
    class_labels = {
        0: "Non détecté (<0.1)",
        1: "Faible (0.1–1.0)",
        2: "Modéré (1.0–10.0)",
        3: "Élevé (>10.0)",
    }
    for cls in [3, 2, 1, 0]:
        mask = gdf_pts["CLASS_CONTAMINATION"] == cls
        if mask.sum() > 0:
            ax.scatter(
                gdf_pts.loc[mask, "X"], gdf_pts.loc[mask, "Y"],
                s=4, color=class_colors[cls], alpha=0.6,
                label=f"{class_labels[cls]} (n={mask.sum():,})",
                rasterized=True,
            )
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    ax.set_title("C — Classes de contamination")
    ax.set_axis_off()

    plt.tight_layout()
    if save:
        out = REPORTS / "05_spatial_contamination_map.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Carte exportée → {out}")

    return fig


def plot_choropleth(
    gdf_communes_agg: "gpd.GeoDataFrame",
    col: str = "taux_median",
    save: bool = True,
) -> plt.Figure:
    """
    Carte choroplèthe du taux médian par commune.
    Nécessite aggregate_by_commune() en amont.
    """
    if not HAS_GEO:
        raise ImportError("geopandas requis.")

    fig, ax = plt.subplots(figsize=(10, 12))
    gdf_communes_agg.plot(
        column=col, ax=ax, cmap="OrRd", legend=True,
        legend_kwds={"label": "Taux médian CLD (mg/kg)", "orientation": "vertical"},
        missing_kwds={"color": "#cccccc", "label": "Pas de données"},
        edgecolor="#666666", linewidth=0.5,
    )
    ax.set_title("Taux Médian de Chlordécone par Commune (Martinique)", fontsize=13)
    ax.set_axis_off()

    plt.tight_layout()
    if save:
        out = REPORTS / "06_choropleth_commune.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Choroplèthe exportée → {out}")

    return fig


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_engineering.ingestion import load_raw
    from data_engineering.cleaning  import clean

    df = clean(load_raw())

    if HAS_GEO:
        gdf_pts = build_geodataframe(df)
        logger.info(f"GeoDataFrame : {len(gdf_pts):,} points")

        try:
            gdf_martinique = load_shapefile("Martinique")
            plot_spatial_map(df, gdf_basemap=gdf_martinique)
        except FileNotFoundError:
            logger.warning("Shapefile Martinique non trouvé, carte sans fond.")
            plot_spatial_map(df)
    else:
        logger.error("geopandas non installé — analyse spatiale impossible.")
