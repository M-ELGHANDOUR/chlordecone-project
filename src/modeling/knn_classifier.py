"""
knn_classifier.py — KNN Classification : détection chlordécone
===============================================================
Cours mobilisé : Data Analysis (27 Fév. Partie 2 — Machine Learning)
Concepts : KNN, validation croisée 5-fold, matrice de confusion, F1-score,
           permutation importance, optimisation de K, Pipeline sklearn

Problème : Classification binaire — une parcelle est-elle contaminée ?
  - Classe 0 : non détecté (CHLORDECONE_RATE ≤ 0.1 mg/kg)
  - Classe 1 : détecté     (CHLORDECONE_RATE > 0.1 mg/kg)

Algorithme KNN :
  Prédiction = vote majoritaire des K plus proches voisins
  Distance euclidienne : d(x,x') = √Σ(xᵢ - xᵢ')²

Usage :
  python src/modeling/knn_classifier.py
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance


REPORTS = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

PALETTE = {"blue": "#2E5FA3", "orange": "#E8531A", "green": "#2D8A5E",
           "red": "#C62828", "grey": "#78909C"}


# ── Sélection du K optimal ─────────────────────────────────────────────────────

def find_best_k(
    X_train: np.ndarray,
    y_train: pd.Series,
    k_range: range = range(1, 31),
    cv: int = 5,
    scoring: str = "f1",
) -> dict:
    """
    Sélectionne le K optimal par validation croisée stratifiée (5-fold).

    Principe : pour chaque K, on entraîne le modèle sur 4/5 des données
    et on évalue sur 1/5. On répète 5 fois (5-fold) et on moyenne les scores.
    Le K qui maximise le F1-score moyen est retenu.

    Returns
    -------
    dict avec best_k, scores, stds
    """
    logger.info(f"  🔍 Recherche du K optimal (range {k_range.start}–{k_range.stop}, cv={cv})...")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores, stds = [], []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        sc = cross_val_score(knn, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1)
        scores.append(sc.mean())
        stds.append(sc.std())

    best_k = k_range[np.argmax(scores)]
    logger.info(f"  ✅ Meilleur K = {best_k} (F1 = {max(scores):.4f})")

    return {
        "k_range": list(k_range),
        "scores":  scores,
        "stds":    stds,
        "best_k":  best_k,
        "best_f1": max(scores),
    }


def plot_k_selection(k_results: dict, save: bool = True) -> plt.Figure:
    """Graphique du F1-score selon K (gauche) + variance (droite)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sélection du K Optimal — KNN Classification", fontsize=13)

    ks      = k_results["k_range"]
    scores  = k_results["scores"]
    stds    = k_results["stds"]
    best_k  = k_results["best_k"]

    # F1-score moyen
    ax = axes[0]
    ax.plot(ks, scores, "o-", color=PALETTE["blue"], linewidth=2, markersize=5)
    ax.fill_between(ks,
                    np.array(scores) - np.array(stds),
                    np.array(scores) + np.array(stds),
                    alpha=0.15, color=PALETTE["blue"])
    ax.axvline(best_k, color=PALETTE["red"], ls="--", lw=1.5, label=f"K optimal = {best_k}")
    ax.set_xlabel("Valeur de K"); ax.set_ylabel("F1-score (CV 5-fold)")
    ax.set_title("F1-score moyen selon K")
    ax.legend()

    # Variance (std)
    ax = axes[1]
    ax.bar(ks, stds, color=PALETTE["grey"], alpha=0.7)
    ax.axvline(best_k, color=PALETTE["red"], ls="--", lw=1.5)
    ax.set_xlabel("Valeur de K"); ax.set_ylabel("Écart-type CV")
    ax.set_title("Variance selon K (stabilité)")

    plt.tight_layout()
    if save:
        out = REPORTS / "07_knn_k_selection.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Figure exportée → {out}")
    return fig


# ── Entraînement et évaluation ────────────────────────────────────────────────

def train_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    best_k: int,
    feature_names: list = None,
) -> dict:
    """
    Entraîne le KNN final et évalue sur le jeu de test.

    Métriques reportées :
    - Accuracy, Precision, Recall, F1 (macro & weighted)
    - Matrice de confusion
    - Rapport de classification complet

    Returns
    -------
    dict avec modèle, métriques et prédictions.
    """
    logger.info(f"  🤖 Entraînement KNN (K={best_k}, metric='euclidean', weights='distance')...")

    # Pipeline : StandardScaler + KNN
    # (X_train est déjà scalé si on passe depuis preprocessing.py)
    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
        weights="distance",   # vote pondéré par 1/distance (meilleur que vote uniforme)
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Métriques
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted")

    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  F1 (weighted) : {f1:.4f}")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")

    return {
        "model":     knn,
        "y_pred":    y_pred,
        "accuracy":  acc,
        "f1":        f1,
        "precision": prec,
        "recall":    rec,
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "cm":        confusion_matrix(y_test, y_pred),
        "feature_names": feature_names,
    }


def plot_confusion_matrix(results: dict, save: bool = True) -> plt.Figure:
    """Affiche la matrice de confusion annotée."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("KNN Classification — Résultats", fontsize=13)

    # Matrice de confusion
    ax = axes[0]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=results["cm"],
        display_labels=["Non détecté (0)", "Détecté (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de Confusion")

    # Métriques bar chart
    ax = axes[1]
    metrics = {
        "Accuracy":  results["accuracy"],
        "Precision": results["precision"],
        "Recall":    results["recall"],
        "F1-score":  results["f1"],
    }
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors)
    for bar, val in zip(bars, metrics.values()):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Métriques de Performance")
    ax.axvline(0.8, color=PALETTE["grey"], ls="--", lw=1, alpha=0.5)

    plt.tight_layout()
    if save:
        out = REPORTS / "08_knn_confusion_metrics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Figure exportée → {out}")
    return fig


# ── Importance des variables ──────────────────────────────────────────────────

def compute_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list,
    n_repeats: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """
    Permutation Importance : mesure l'impact de chaque variable sur la performance.

    Principe : on mélange aléatoirement les valeurs d'une variable (20 répétitions).
    La dégradation du F1-score = importance de cette variable.
    Variables avec importance ≈ 0 → peu ou pas d'effet sur le modèle.
    """
    logger.info(f"  🔬 Calcul de la permutation importance ({n_repeats} répétitions)...")

    pi = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
        scoring="f1_weighted",
    )

    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": pi.importances_mean,
        "std":        pi.importances_std,
    }).sort_values("importance", ascending=False)

    logger.info(f"\n  Top 3 variables :\n{importance_df.head(3).to_string(index=False)}")

    # Visualisation
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["orange"] if i < 3 else PALETTE["blue"]
              for i in range(len(importance_df))]
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        xerr=importance_df["std"][::-1],
        color=colors[::-1], alpha=0.85, capsize=4,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Dégradation F1 (importance)")
    ax.set_title("Permutation Importance — KNN Classification")
    plt.tight_layout()

    if save:
        out = REPORTS / "09_feature_importance.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Figure exportée → {out}")

    return importance_df


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_engineering.ingestion import load_raw
    from data_engineering.cleaning  import clean
    from modeling.preprocessing     import prepare_full

    df   = clean(load_raw())
    data = prepare_full(df, task="classification", encode_soil=True)

    # 1 — Sélection du K optimal
    k_results = find_best_k(data["X_train"], data["y_train"], k_range=range(1, 31))
    plot_k_selection(k_results)

    # 2 — Entraînement et évaluation
    results = train_evaluate(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        best_k=k_results["best_k"],
        feature_names=data["feature_names"],
    )
    plot_confusion_matrix(results)

    # 3 — Importance des variables
    importance = compute_feature_importance(
        results["model"],
        data["X_test"], data["y_test"],
        data["feature_names"],
    )
    print(f"\n── Feature Importance ─────────────────────────────────")
    print(importance.to_string(index=False))
