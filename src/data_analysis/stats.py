"""
stats.py — Analyse statistique complète
=======================================
Cours mobilisé : Data Analysis (Séances 05 & 12 Janv., 23 & 30 Janv., 27 Fév., 02 Mars)
Concepts : statistiques descriptives, tests non-paramétriques, corrélations,
           loi log-normale, Kruskal-Wallis, Mann-Whitney, χ², Spearman, Pearson

Fonctions :
  - descriptive_stats()    : stats descriptives enrichies
  - test_log_normality()   : KS + Shapiro-Wilk sur log(CLD)
  - kruskal_wallis_test()  : test non-paramétrique multi-groupes
  - mann_whitney_test()    : comparaison deux groupes
  - chi2_independence()    : test d'indépendance χ²
  - spearman_correlation() : corrélation de rang
  - correlation_matrix()   : matrice Pearson & Spearman
"""

import pandas as pd
import numpy as np
from scipy.stats import (
    kstest, shapiro, kruskal, mannwhitneyu,
    spearmanr, pearsonr, chi2_contingency, norm
)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Statistiques descriptives ─────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, target: str = "CHLORDECONE_RATE") -> pd.DataFrame:
    """
    Statistiques descriptives enrichies sur la variable cible.

    Inclut : moyenne, médiane, écart-type, skewness, kurtosis, quantiles.
    """
    col = df[target].dropna()

    stats = pd.Series({
        "n":                len(col),
        "n_missing":        df[target].isna().sum(),
        "mean":             col.mean(),
        "median":           col.median(),
        "std":              col.std(),
        "min":              col.min(),
        "Q1 (25%)":         col.quantile(0.25),
        "Q3 (75%)":         col.quantile(0.75),
        "max":              col.max(),
        "IQR":              col.quantile(0.75) - col.quantile(0.25),
        "skewness":         col.skew(),
        "kurtosis":         col.kurtosis(),
        "% > 0.1 (détecté)": (col > 0.1).mean() * 100,
        "% > 1.0 (réglementaire)": (col > 1.0).mean() * 100,
    }, name=target)

    return stats.to_frame()


# ── Test de loi log-normale ───────────────────────────────────────────────────

def test_log_normality(
    df: pd.DataFrame,
    col: str = "CHLORDECONE_RATE",
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict:
    """
    Teste si log(1 + col) suit une loi normale (hypothèse log-normale).

    Tests utilisés :
    - Kolmogorov-Smirnov (KS) : compare la distribution empirique à N(μ, σ)
    - Shapiro-Wilk            : test de normalité (exact, < 5000 obs.)

    Returns
    -------
    dict avec statistiques et p-values.
    """
    detected = df[df[col] > 0.1][col].dropna()
    log_vals = np.log1p(detected)

    # Sous-échantillon pour Shapiro (limité à 5000)
    sample = log_vals.sample(min(sample_size, len(log_vals)), random_state=random_state)

    # KS test
    ks_stat, ks_p = kstest(sample, "norm", args=(sample.mean(), sample.std()))

    # Shapiro-Wilk
    sw_stat, sw_p = shapiro(sample.sample(min(5000, len(sample)), random_state=0))

    result = {
        "n_sample":    len(sample),
        "log_mean":    sample.mean(),
        "log_std":     sample.std(),
        "KS_statistic": round(ks_stat, 4),
        "KS_pvalue":    round(ks_p, 4),
        "SW_statistic": round(sw_stat, 4),
        "SW_pvalue":    round(sw_p, 6),
        "conclusion":   "Log-normale ✓" if ks_p > 0.05 else "Non log-normale ✗",
    }

    logger.info(f"  KS test  : stat={ks_stat:.4f}, p={ks_p:.4f}")
    logger.info(f"  SW test  : stat={sw_stat:.4f}, p={sw_p:.6f}")
    logger.info(f"  → {result['conclusion']}")

    return result


# ── Kruskal-Wallis ────────────────────────────────────────────────────────────

def kruskal_wallis_test(
    df: pd.DataFrame,
    group_col: str = "SOL_SIMPLE",
    value_col: str = "CHLORDECONE_RATE",
    min_group_size: int = 30,
) -> dict:
    """
    Test de Kruskal-Wallis : les médianes sont-elles égales entre groupes ?

    Hypothèses :
    - H₀ : Les médianes sont égales entre tous les groupes
    - H₁ : Au moins une médiane est différente

    Équivalent non-paramétrique de l'ANOVA à un facteur.
    Utilise les rangs plutôt que les valeurs brutes → robuste à la non-normalité.
    """
    groups = [
        grp[value_col].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp) >= min_group_size
    ]

    h_stat, p_val = kruskal(*groups)

    result = {
        "test":         "Kruskal-Wallis",
        "n_groups":     len(groups),
        "H_statistic":  round(h_stat, 4),
        "p_value":      p_val,
        "significant":  p_val < 0.05,
        "interpretation": (
            "✅ Différences significatives entre groupes (p < 0.05)"
            if p_val < 0.05
            else "❌ Pas de différence significative détectée"
        ),
    }

    logger.info(f"  Kruskal-Wallis H={h_stat:.4f}, p={p_val:.2e}")
    logger.info(f"  → {result['interpretation']}")
    return result


# ── Mann-Whitney ──────────────────────────────────────────────────────────────

def mann_whitney_test(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    value_col: str = "CHLORDECONE_RATE",
) -> dict:
    """
    Test de Mann-Whitney U : comparaison de deux groupes indépendants.
    Équivalent non-paramétrique du t-test de Student.
    """
    a = df[df[group_col] == group_a][value_col].dropna().values
    b = df[df[group_col] == group_b][value_col].dropna().values

    u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")

    result = {
        "test":          "Mann-Whitney U",
        "group_A":       group_a,
        "group_B":       group_b,
        "n_A":           len(a),
        "n_B":           len(b),
        "median_A":      np.median(a),
        "median_B":      np.median(b),
        "U_statistic":   round(u_stat, 2),
        "p_value":       p_val,
        "significant":   p_val < 0.05,
    }
    return result


# ── Chi-2 ─────────────────────────────────────────────────────────────────────

def chi2_independence(
    df: pd.DataFrame,
    col_a: str = "SOL_SIMPLE",
    col_b: str = "CLASS_CONTAMINATION",
) -> dict:
    """
    Test du χ² d'indépendance entre deux variables catégorielles.

    H₀ : les deux variables sont indépendantes.
    Mesure complémentaire : V de Cramér (force de l'association, 0–1).
    """
    contingency = pd.crosstab(
        df[col_a].fillna("Unknown"),
        df[col_b].astype(str),
    )

    chi2_stat, p_chi2, dof, _ = chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2_stat / (len(df) * (min(contingency.shape) - 1)))

    result = {
        "test":        "Chi-2 d'indépendance",
        "chi2":        round(chi2_stat, 4),
        "p_value":     p_chi2,
        "dof":         dof,
        "cramers_v":   round(cramers_v, 4),
        "significant": p_chi2 < 0.05,
        "association": (
            "Forte" if cramers_v > 0.3
            else "Modérée" if cramers_v > 0.1
            else "Faible"
        ),
    }

    logger.info(f"  χ²={chi2_stat:.2f}, p={p_chi2:.2e}, V de Cramér={cramers_v:.3f}")
    return result


# ── Corrélations ──────────────────────────────────────────────────────────────

def spearman_correlation(
    df: pd.DataFrame,
    col_x: str = "MNT_SLOPE_MEAN",
    col_y: str = "CHLORDECONE_RATE",
) -> dict:
    """
    Corrélation de Spearman (rang) — robuste aux distributions asymétriques.
    Complétée par la corrélation de Pearson (linéaire) pour comparaison.
    """
    valid = df[[col_x, col_y]].dropna()
    rho, p_rho   = spearmanr(valid[col_x], valid[col_y])
    r, p_pearson = pearsonr(valid[col_x], valid[col_y])

    return {
        "Spearman_rho": round(rho, 4),
        "Spearman_p":   p_rho,
        "Pearson_r":    round(r, 4),
        "Pearson_p":    p_pearson,
        "n":            len(valid),
        "note": (
            "Spearman préféré (données non-normales, asymétriques)"
        ),
    }


def correlation_matrix(
    df: pd.DataFrame,
    features: list,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Matrice de corrélation (Pearson ou Spearman) sur les variables numériques.
    """
    return df[features].corr(method=method)


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from data_engineering.ingestion import load_raw
    from data_engineering.cleaning  import clean

    df = clean(load_raw())

    print("\n── Statistiques descriptives ──────────────────────────")
    print(descriptive_stats(df).to_string())

    print("\n── Test log-normalité ─────────────────────────────────")
    res = test_log_normality(df)
    for k, v in res.items():
        print(f"  {k}: {v}")

    print("\n── Kruskal-Wallis (Sol × Taux CLD) ────────────────────")
    res2 = kruskal_wallis_test(df)
    for k, v in res2.items():
        print(f"  {k}: {v}")

    print("\n── Chi-2 (Sol × Classe contamination) ─────────────────")
    res3 = chi2_independence(df)
    for k, v in res3.items():
        print(f"  {k}: {v}")
