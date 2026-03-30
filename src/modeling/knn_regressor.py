"""
knn_regressor.py — KNN Régression : prédiction du taux de chlordécone
======================================================================
Cours mobilisé : Data Analysis (27 Fév. Partie 2 — Machine Learning)
Concepts : KNN régression, RMSE, R², courbe biais-variance,
           prédiction vs valeurs réelles, interprétation des résidus

Problème : Régression — prédire le log(1 + Taux_CLD) d'une parcelle
  à partir de ses caractéristiques topographiques et pédologiques.

Différence KNN Classification vs Régression :
  - Classification : vote majoritaire → classe prédite
  - Régression     : moyenne pondérée des valeurs des K voisins → valeur continue

Usage :
  python src/modeling/knn_regressor.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


REPORTS = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

PALETTE = {"blue": "#2E5FA3", "orange": "#E8531A", "green": "#2D8A5E",
           "red": "#C62828", "grey": "#78909C"}


# ── Sélection du K optimal (régression) ───────────────────────────────────────

def find_best_k_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    k_range: range = range(3, 31, 2),
    cv: int = 5,
) -> dict:
    """
    Sélectionne le K minimisant le RMSE (validation croisée 5-fold).

    Méthode : on utilise neg_root_mean_squared_error — sklearn minimise,
    donc on cherche le maximum de neg_RMSE (équivalent au minimum de RMSE).

    Returns
    -------
    dict avec best_k, rmse_train, rmse_test
    """
    logger.info(f"  🔍 Régression — Recherche K optimal (range {k_range.start}–{k_range.stop})...")

    cv_strategy  = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmse_cv, stds = [], []
    rmse_train    = []

    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)

        # RMSE sur test (CV)
        sc = cross_val_score(
            knn, X_train, y_train,
            cv=cv_strategy, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        rmse_cv.append(-sc.mean())
        stds.append(sc.std())

        # RMSE sur train (pour courbe biais-variance)
        knn.fit(X_train, y_train)
        y_pred_train = knn.predict(X_train)
        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    best_k    = k_range[np.argmin(rmse_cv)]
    best_rmse = min(rmse_cv)
    logger.info(f"  ✅ Meilleur K = {best_k} (RMSE CV = {best_rmse:.4f})")

    return {
        "k_range":    list(k_range),
        "rmse_cv":    rmse_cv,
        "rmse_train": rmse_train,
        "stds":       stds,
        "best_k":     best_k,
        "best_rmse":  best_rmse,
    }


def plot_k_regression(k_results: dict, save: bool = True) -> plt.Figure:
    """Courbes RMSE train vs test selon K — visualise le biais-variance trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sélection du K Optimal — KNN Régression", fontsize=13)

    ks    = k_results["k_range"]
    best_k = k_results["best_k"]

    # RMSE train vs CV
    ax = axes[0]
    ax.plot(ks, k_results["rmse_train"], "s--", color=PALETTE["orange"],
            linewidth=1.5, markersize=5, label="RMSE train")
    ax.plot(ks, k_results["rmse_cv"],   "o-",  color=PALETTE["blue"],
            linewidth=2,   markersize=5, label="RMSE test (CV)")
    ax.fill_between(ks,
                    np.array(k_results["rmse_cv"]) - np.array(k_results["stds"]),
                    np.array(k_results["rmse_cv"]) + np.array(k_results["stds"]),
                    alpha=0.15, color=PALETTE["blue"])
    ax.axvline(best_k, color=PALETTE["red"], ls="--", lw=1.5, label=f"K = {best_k}")
    ax.set_xlabel("K"); ax.set_ylabel("RMSE (log-taux)")
    ax.set_title("Trade-off Biais–Variance")
    ax.legend()

    # Détail autour du meilleur K
    ax = axes[1]
    ax.bar(ks, k_results["rmse_cv"], color=PALETTE["blue"], alpha=0.7)
    ax.axvline(best_k, color=PALETTE["red"], ls="--", lw=2, label=f"K optimal = {best_k}")
    ax.set_xlabel("K"); ax.set_ylabel("RMSE (CV)")
    ax.set_title("RMSE par K")
    ax.legend()

    plt.tight_layout()
    if save:
        out = REPORTS / "10_knn_regression_k_selection.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Figure exportée → {out}")
    return fig


# ── Entraînement et évaluation ────────────────────────────────────────────────

def train_evaluate_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    best_k: int,
) -> dict:
    """
    Entraîne le KNN régression final et évalue sur le jeu de test.

    Métriques :
    - RMSE : racine de l'erreur quadratique moyenne (même unité que log-taux)
    - MAE  : erreur absolue moyenne (robuste aux outliers)
    - R²   : coefficient de détermination (variance expliquée)

    Returns
    -------
    dict avec modèle, métriques, prédictions.
    """
    logger.info(f"  🤖 Entraînement KNN Régression (K={best_k})...")

    knn = KNeighborsRegressor(
        n_neighbors=best_k,
        weights="distance",   # les voisins proches pèsent plus
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    logger.info(f"  RMSE : {rmse:.4f}")
    logger.info(f"  MAE  : {mae:.4f}")
    logger.info(f"  R²   : {r2:.4f}")

    # Interprétation RMSE en échelle originale
    rmse_original = np.expm1(rmse) - 1
    logger.info(
        f"  Interprétation : RMSE ≈ {rmse:.3f} sur log-taux "
        f"→ erreur ≈ ×{np.exp(rmse):.2f} en échelle originale"
    )

    return {
        "model":  knn,
        "y_pred": y_pred,
        "rmse":   rmse,
        "mae":    mae,
        "r2":     r2,
    }


def plot_regression_results(
    results: dict,
    y_test: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Graphiques d'évaluation : prédictions vs réel, résidus, distribution des erreurs."""
    y_pred = results["y_pred"]
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"KNN Régression — RMSE={results['rmse']:.3f} | MAE={results['mae']:.3f} | R²={results['r2']:.3f}",
        fontsize=13,
    )

    # A — Prédictions vs valeurs réelles
    ax = axes[0]
    ax.scatter(y_test, y_pred, s=4, alpha=0.3, color=PALETTE["blue"], rasterized=True)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Parfait (y=x)")
    ax.set_xlabel("Valeurs réelles log(1+CLD)")
    ax.set_ylabel("Valeurs prédites log(1+CLD)")
    ax.set_title("A — Prédit vs Réel")
    ax.legend()

    # B — Résidus vs prédictions
    ax = axes[1]
    ax.scatter(y_pred, residuals, s=4, alpha=0.3, color=PALETTE["orange"], rasterized=True)
    ax.axhline(0, color=PALETTE["red"], ls="--", lw=1.5)
    ax.axhline( results["rmse"], color=PALETTE["grey"], ls=":", lw=1, alpha=0.7)
    ax.axhline(-results["rmse"], color=PALETTE["grey"], ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Valeurs prédites")
    ax.set_ylabel("Résidus (réel − prédit)")
    ax.set_title("B — Résidus vs Prédictions")

    # C — Distribution des résidus (normalité ?)
    ax = axes[2]
    ax.hist(residuals, bins=60, color=PALETTE["green"], edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=PALETTE["red"], ls="--", lw=1.5)
    ax.axvline( results["rmse"], color=PALETTE["grey"], ls=":", lw=1, label=f"±RMSE ({results['rmse']:.3f})")
    ax.axvline(-results["rmse"], color=PALETTE["grey"], ls=":", lw=1)
    ax.set_xlabel("Résidu")
    ax.set_ylabel("Fréquence")
    ax.set_title("C — Distribution des résidus")
    ax.legend()

    plt.tight_layout()
    if save:
        out = REPORTS / "11_knn_regression_results.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Figure exportée → {out}")
    return fig


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_engineering.ingestion import load_raw
    from data_engineering.cleaning  import clean
    from modeling.preprocessing     import prepare_full

    df   = clean(load_raw())
    data = prepare_full(df, task="regression", encode_soil=True)

    # 1 — Sélection K optimal
    k_results = find_best_k_regression(data["X_train"], data["y_train"])
    plot_k_regression(k_results)

    # 2 — Entraînement et évaluation
    results = train_evaluate_regression(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        best_k=k_results["best_k"],
    )
    plot_regression_results(results, data["y_test"])

    print(f"\n── Résultats Régression ───────────────────────────────")
    print(f"  K optimal : {k_results['best_k']}")
    print(f"  RMSE      : {results['rmse']:.4f}")
    print(f"  MAE       : {results['mae']:.4f}")
    print(f"  R²        : {results['r2']:.4f}")
