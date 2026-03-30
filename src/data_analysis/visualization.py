"""
visualization.py — Graphiques et visualisations
================================================
Cours mobilisé : Data Analysis (Séance 02 Mars, 27 Fév.)
                 Data Engineering (Séance 09 Mars — Cartographie Python)
Concepts : matplotlib, seaborn, distributions, corrélations, analyse temporelle

Fonctions :
  - plot_distribution()       : distribution du taux CLD (6 panels)
  - plot_correlation_matrix() : heatmap Pearson & Spearman
  - plot_temporal_trend()     : évolution 2010-2019
  - plot_top_communes()       : bar chart communes les plus contaminées
  - plot_soil_boxplots()      : boxplots par type de sol
  - save_fig()                : exportation propre vers reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend non-interactif (pour scripts CI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Palette et style global ───────────────────────────────────────────────────
PALETTE = {
    "orange":  "#E8531A",
    "green":   "#2D8A5E",
    "blue":    "#2E5FA3",
    "red":     "#C62828",
    "grey":    "#78909C",
    "yellow":  "#F9A825",
    "purple":  "#6A1B9A",
    "light":   "#F5F5F5",
}

CMAP_POLLUTION = LinearSegmentedColormap.from_list(
    "pollution",
    ["#2D8A5E", "#F9A825", "#E8531A", "#C62828"],
    N=256,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """Exporte une figure dans reports/figures/."""
    out = REPORTS_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info(f"📊 Figure exportée → {out}")
    return out


# ── Distribution du taux CLD ──────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    6 panels illustrant la distribution du taux de chlordécone :
      A — Histogramme brut (détectés)
      B — Transformation log(1+x)
      C — Q-Q plot (log-normalité)
      D — CDF empirique vs théorique
      E — Boxplots par type de sol
      F — KDE par type de sol
    """
    detected = df[df["DETECTED"] == 1]["CHLORDECONE_RATE"]
    log_vals  = np.log1p(detected)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Distribution du Taux de Chlordécone dans les Sols Antillais",
        fontsize=16, fontweight="bold", y=1.02,
    )

    # A — Histogramme brut
    ax = axes[0, 0]
    ax.hist(detected, bins=80, color=PALETTE["orange"], edgecolor="white", linewidth=0.3)
    ax.set_title("A — Distribution brute (détectés)")
    ax.set_xlabel("Taux CLD (mg/kg)"); ax.set_ylabel("Fréquence")
    ax.axvline(1.0, color=PALETTE["red"], ls="--", lw=1.5, label="Seuil régl. 1.0")
    ax.legend(fontsize=9)

    # B — Log-transformation
    ax = axes[0, 1]
    ax.hist(log_vals, bins=80, color=PALETTE["blue"], edgecolor="white", linewidth=0.3)
    ax.set_title("B — Distribution log(1 + CLD)")
    ax.set_xlabel("log(1 + Taux CLD)"); ax.set_ylabel("Fréquence")

    # C — Q-Q plot
    from scipy.stats import probplot
    ax = axes[0, 2]
    res = probplot(log_vals, dist="norm")
    ax.scatter(res[0][0], res[0][1], s=3, alpha=0.3, color=PALETTE["blue"])
    ax.plot(res[0][0], res[1][0]*res[0][0]+res[1][1], color=PALETTE["red"], lw=1.5)
    ax.set_title("C — Q-Q plot (log-normalité)")
    ax.set_xlabel("Quantiles théoriques N(0,1)")
    ax.set_ylabel("Quantiles empiriques log(CLD)")

    # D — CDF empirique vs théorique
    from scipy.stats import norm as sp_norm
    ax = axes[1, 0]
    sorted_vals = np.sort(log_vals)
    cdf_emp = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    cdf_th  = sp_norm.cdf(sorted_vals, sorted_vals.mean(), sorted_vals.std())
    ax.plot(sorted_vals, cdf_emp, label="CDF empirique", color=PALETTE["blue"])
    ax.plot(sorted_vals, cdf_th, label="CDF N(μ,σ)", color=PALETTE["red"], ls="--")
    ax.set_title("D — CDF empirique vs théorique")
    ax.set_xlabel("log(1 + CLD)"); ax.set_ylabel("Probabilité cumulée")
    ax.legend(fontsize=9)

    # E — Boxplots par sol (triés par médiane décroissante)
    ax = axes[1, 1]
    sol_order = (df.groupby("SOL_SIMPLE")["CHLORDECONE_RATE"]
                   .median().sort_values(ascending=False).index.tolist())
    data_by_sol = [
        df[df["SOL_SIMPLE"] == sol]["CHLORDECONE_RATE"].dropna().values
        for sol in sol_order
    ]
    bp = ax.boxplot(data_by_sol, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["orange"])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(sol_order) + 1))
    ax.set_xticklabels(sol_order, rotation=30, ha="right", fontsize=7)
    ax.set_title("E — Boxplots par type de sol")
    ax.set_xlabel("Type de sol"); ax.set_ylabel("Taux CLD (mg/kg)")
    ax.set_yscale("log")

    # F — KDE par sol
    ax = axes[1, 2]
    colors_sol = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
    for i, (sol, grp) in enumerate(df.groupby("SOL_SIMPLE")):
        vals = np.log1p(grp["CHLORDECONE_RATE"].dropna())
        if len(vals) > 100:
            vals.plot.kde(ax=ax, label=sol, color=colors_sol[i % len(colors_sol)], linewidth=1.5)
    ax.set_title("F — KDE log(CLD) par type de sol")
    ax.set_xlabel("log(1 + Taux CLD)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    if save:
        save_fig(fig, "01_distribution_taux_cld.png")
    return fig


# ── Matrice de corrélation ────────────────────────────────────────────────────

def plot_correlation_matrix(
    df: pd.DataFrame,
    features: list = None,
    save: bool = True,
) -> plt.Figure:
    """Heatmap des corrélations Pearson et Spearman côte à côte."""
    if features is None:
        features = [
            "CHLORDECONE_RATE", "RATE_5B_HYDRO", "HISTOBANANE_HISTO_BAN",
            "MNT_TPI_MEAN", "MNT_TRI_MEAN", "MNT_RUGOSITY_MEAN",
            "MNT_SHADING_MEAN", "MNT_EXPOSURE_MEAN", "MNT_SLOPE_MEAN",
        ]
    features = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Matrices de Corrélation — Chlordécone & Variables Topographiques",
                 fontsize=14, fontweight="bold")

    for ax, method in zip(axes, ["pearson", "spearman"]):
        corr = df[features].corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.5, square=True,
        )
        ax.set_title(f"Corrélation de {method.capitalize()}", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    if save:
        save_fig(fig, "02_correlation_matrix.png")
    return fig


# ── Tendance temporelle ───────────────────────────────────────────────────────

def plot_temporal_trend(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Évolution annuelle du taux médian, détections et variabilité (2010-2019)."""
    yearly = df.groupby("YEAR")["CHLORDECONE_RATE"].agg(
        ["median", "mean", "count", "std"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Évolution Temporelle de la Contamination (2010–2019)",
                 fontsize=14, fontweight="bold")

    # A — Médiane et moyenne annuelles
    ax = axes[0, 0]
    ax.fill_between(yearly.index, yearly["median"] - yearly["std"]/2,
                    yearly["median"] + yearly["std"]/2,
                    alpha=0.15, color=PALETTE["orange"])
    ax.plot(yearly.index, yearly["median"], "o-", color=PALETTE["orange"],
            linewidth=2, markersize=7, label="Médiane")
    ax.plot(yearly.index, yearly["mean"],   "s--", color=PALETTE["blue"],
            linewidth=1.5, markersize=5, label="Moyenne")
    ax.set_title("A — Taux médian et moyen annuels")
    ax.set_xlabel("Année"); ax.set_ylabel("Taux CLD (mg/kg)")
    ax.legend()

    # B — Nombre de prélèvements par année
    ax = axes[0, 1]
    ax.bar(yearly.index, yearly["count"], color=PALETTE["blue"], alpha=0.8)
    ax.set_title("B — Nombre de prélèvements par année")
    ax.set_xlabel("Année"); ax.set_ylabel("N prélèvements")

    # C — % de détections par année
    ax = axes[1, 0]
    pct_detected = df.groupby("YEAR")["DETECTED"].mean() * 100
    ax.plot(pct_detected.index, pct_detected.values, "o-",
            color=PALETTE["green"], linewidth=2, markersize=7)
    ax.axhline(pct_detected.mean(), color=PALETTE["grey"], ls="--", lw=1.5)
    ax.set_title("C — % Détections par année")
    ax.set_xlabel("Année"); ax.set_ylabel("% Détecté > 0.1 mg/kg")
    ax.set_ylim(0, 100)

    # D — Boxplots par année
    ax = axes[1, 1]
    years = sorted(df["YEAR"].dropna().unique())
    data_by_year = [df[df["YEAR"] == y]["CHLORDECONE_RATE"].dropna().values for y in years]
    bp = ax.boxplot(data_by_year, labels=years, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=PALETTE["orange"], alpha=0.6))
    ax.set_title("D — Distribution annuelle (sans outliers)")
    ax.set_xlabel("Année"); ax.set_ylabel("Taux CLD (mg/kg)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        save_fig(fig, "03_temporal_trend.png")
    return fig


# ── Top communes ──────────────────────────────────────────────────────────────

def plot_top_communes(df: pd.DataFrame, top_n: int = 20, save: bool = True) -> plt.Figure:
    """Bar chart des communes avec le taux médian le plus élevé."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Analyse par Commune — Contamination Chlordécone",
                 fontsize=14, fontweight="bold")

    # Panel A — Taux médian
    ax = axes[0]
    top_med = (df.groupby("COMMU_LAB")["CHLORDECONE_RATE"]
               .median().sort_values(ascending=False).head(top_n))
    colors = [PALETTE["red"] if v > 1.0 else PALETTE["orange"] for v in top_med.values]
    ax.barh(top_med.index[::-1], top_med.values[::-1], color=colors[::-1])
    ax.axvline(1.0, color=PALETTE["red"], ls="--", lw=1.5, label="Seuil régl. 1.0 mg/kg")
    ax.axvline(0.1, color=PALETTE["yellow"], ls=":", lw=1.5, label="Seuil détection 0.1 mg/kg")
    ax.set_title(f"Taux médian par commune (Top {top_n})")
    ax.set_xlabel("Taux médian CLD (mg/kg)")
    ax.legend(fontsize=9)

    # Panel B — % au-dessus du seuil réglementaire
    ax = axes[1]
    pct_above = (df.groupby("COMMU_LAB")["ABOVE_REG_THRESHOLD"]
                 .mean().mul(100).sort_values(ascending=False).head(top_n))
    ax.barh(pct_above.index[::-1], pct_above.values[::-1], color=PALETTE["purple"])
    ax.set_title(f"% Dépassement seuil 1.0 mg/kg (Top {top_n})")
    ax.set_xlabel("% des prélèvements > 1.0 mg/kg")

    plt.tight_layout()
    if save:
        save_fig(fig, "04_top_communes.png")
    return fig


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_engineering.ingestion import load_raw
    from data_engineering.cleaning  import clean

    df = clean(load_raw())

    plot_distribution(df)
    plot_correlation_matrix(df)
    plot_temporal_trend(df)
    plot_top_communes(df)
    logger.info("✅ Toutes les figures générées dans reports/figures/")
