"""
preprocessing.py — Préparation des features pour la modélisation KNN
=====================================================================
Cours mobilisé : Data Analysis (27 Fév. — Partie 2 ML), Data Science
Concepts : feature engineering, encodage, StandardScaler, train/test split,
           validation croisée, gestion des déséquilibres de classes

Fonctions :
  - build_feature_matrix()    : sélection et encodage des variables
  - split_data()              : train/test split stratifié
  - scale_features()          : standardisation Z-score (StandardScaler)
  - get_class_weights()       : poids des classes (déséquilibre)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Variables utilisées pour la modélisation ──────────────────────────────────
FEATURES_BASE = [
    "MNT_SLOPE_MEAN",          # Pente (°)
    "MNT_TPI_MEAN",            # Topographic Position Index
    "MNT_TRI_MEAN",            # Terrain Ruggedness Index
    "MNT_EXPOSURE_MEAN",       # Exposition (°)
    "MNT_SHADING_MEAN",        # Ombrage
    "MNT_RUGOSITY_MEAN",       # Rugosité
    "HISTOBANANE_HISTO_BAN",   # Périodes historiques de bananiers
]

FEATURES_EXTENDED = FEATURES_BASE + [
    "SOL_SIMPLE_ENCODED",      # Type de sol encodé (label encoding)
]

# Variables cibles
TARGET_CLASS = "DETECTED"              # Classification binaire
TARGET_REG   = "LOG_CHLD"             # Régression (log-taux)
TARGET_CLASS4 = "CLASS_CONTAMINATION" # Classification 4 classes


def build_feature_matrix(
    df: pd.DataFrame,
    features: list = None,
    encode_soil: bool = True,
) -> pd.DataFrame:
    """
    Construit la matrice de features X pour la modélisation.

    Inclut optionnellement le type de sol encodé numériquement.
    Gère les valeurs manquantes par imputation par la médiane.

    Parameters
    ----------
    df : pd.DataFrame nettoyé
    features : liste de variables (défaut : FEATURES_BASE)
    encode_soil : ajouter SOL_SIMPLE encodé en entier

    Returns
    -------
    pd.DataFrame — matrice X prête pour sklearn
    """
    if features is None:
        features = FEATURES_BASE.copy()

    X = df[features].copy()

    # Encodage du type de sol
    if encode_soil and "SOL_SIMPLE" in df.columns:
        le = LabelEncoder()
        X["SOL_SIMPLE_ENCODED"] = le.fit_transform(
            df["SOL_SIMPLE"].fillna("Unknown")
        )
        logger.info(f"  🔢 SOL_SIMPLE encodé → {le.classes_}")

    # Imputation des valeurs manquantes par la médiane
    for col in X.columns:
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info(f"  ⚕️  {col} : {n_missing} NaN → médiane ({median_val:.4f})")

    logger.info(f"  ✅ Matrice X : {X.shape[0]:,} obs × {X.shape[1]} features")
    return X


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple:
    """
    Split train/test stratifié (conserve la proportion des classes).

    Parameters
    ----------
    X : matrice de features
    y : variable cible
    test_size : proportion pour le test (défaut 20%)
    stratify : True pour la classification (conserver les proportions)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    logger.info(f"  📊 Train : {len(X_tr):,} obs | Test : {len(X_te):,} obs")
    if stratify:
        logger.info(
            f"  Distribution (train) : {y_tr.value_counts(normalize=True).to_dict()}"
        )

    return X_tr, X_te, y_tr, y_te


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Standardisation Z-score : X' = (X - μ) / σ

    IMPORTANT pour KNN : l'algorithme est basé sur des distances euclidiennes.
    Sans standardisation, une variable avec de grandes valeurs numériques
    (ex. MNT_SHADING_MEAN ≈ 100) domine les distances face à une variable
    centrée sur 0 (ex. MNT_TPI_MEAN ≈ 0).

    Le scaler est entraîné UNIQUEMENT sur le train pour éviter le data leakage.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler (pour réutilisation en production)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform sur train
    X_test_scaled  = scaler.transform(X_test)         # transform seulement sur test

    logger.info(f"  ⚖️  StandardScaler : μ={scaler.mean_.round(3)[:3]}... σ={scaler.scale_.round(3)[:3]}...")
    return X_train_scaled, X_test_scaled, scaler


def get_class_weights(y: pd.Series) -> dict:
    """
    Calcule les poids inverses des classes pour gérer le déséquilibre.

    Formule : w_c = n_total / (n_classes × n_c)

    Returns
    -------
    dict {classe: poids}
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    logger.info(f"  ⚖️  Poids des classes : {weight_dict}")
    return weight_dict


def prepare_full(
    df: pd.DataFrame,
    task: str = "classification",
    encode_soil: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Préparation complète : X, y, split, scaling.

    Parameters
    ----------
    task : "classification" (DETECTED), "regression" (LOG_CHLD),
           "multiclass" (CLASS_CONTAMINATION)

    Returns
    -------
    dict avec X_tr, X_te, y_tr, y_te, scaler, feature_names
    """
    logger.info(f"  🎯 Tâche : {task}")

    # Sélection de la cible
    target_map = {
        "classification": TARGET_CLASS,
        "regression":     TARGET_REG,
        "multiclass":     TARGET_CLASS4,
    }
    target_col = target_map.get(task, TARGET_CLASS)

    # Construction de X
    X = build_feature_matrix(df, encode_soil=encode_soil)
    y = df[target_col].dropna()

    # Alignement des index
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Split
    stratify = (task != "regression")
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size, random_state, stratify)

    # Scaling
    X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)

    return {
        "X_train":       X_tr_sc,
        "X_test":        X_te_sc,
        "y_train":       y_tr,
        "y_test":        y_te,
        "scaler":        scaler,
        "feature_names": list(X.columns),
        "target":        target_col,
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
    }


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from data_engineering.ingestion import load_raw
    from data_engineering.cleaning  import clean

    df = clean(load_raw())
    data = prepare_full(df, task="classification")

    print(f"\n── Préparation complète ────────────────────────────────")
    print(f"  Features   : {data['feature_names']}")
    print(f"  Cible      : {data['target']}")
    print(f"  Train      : {data['n_train']:,} obs.")
    print(f"  Test       : {data['n_test']:,} obs.")
    print(f"  X_train shape : {data['X_train'].shape}")
