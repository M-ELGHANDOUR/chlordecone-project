"""
cleaning.py — Nettoyage et transformation des données
======================================================
Cours mobilisé : Data Engineering (Séance 12 & 26 Janv.), Data Analysis (02 Mars)
Concepts : ETL (Transform), feature engineering, encodage, gestion des outliers

Étapes :
  1. Renommage cohérent avec le dictionnaire des variables
  2. Conversion des types (dates, numériques)
  3. Nettoyage du taux 5b (valeurs texte → float)
  4. Feature engineering (DETECTED, LOG_CHLD, CLASS_CONTAMINATION, etc.)
  5. Traitement des valeurs manquantes
"""

import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Mapping renommage ─────────────────────────────────────────────────────────
RENAME_MAP = {
    "ANNEE":                  "YEAR",
    "Sol_simple":             "SOL_SIMPLE",
    "type_sol":               "TYPE_SOL",
    "Date_prelevement":       "DATE_OF_SAMPLING",
    "Date_enregistrement":    "DATE_RECORDED",
    "Date_analyse":           "DATE_ANALYSIS",
    "Operateur_chld":         "OPERATOR_CLD",
    "Taux_Chlordecone":       "CHLORDECONE_RATE",
    "Operateur_5b":           "OPERATOR_5B",
    "Taux_5b_hydro":          "RATE_5B_HYDRO",
    "histoBanane_Histo_ban":  "HISTOBANANE_HISTO_BAN",
    "mnt_tpi_mean":           "MNT_TPI_MEAN",
    "mnt_tri_mean":           "MNT_TRI_MEAN",
    "mnt_rugosite_mean":      "MNT_RUGOSITY_MEAN",
    "mnt_ombrage_mean":       "MNT_SHADING_MEAN",
    "mnt_exposition_mean":    "MNT_EXPOSURE_MEAN",
    "mnt_pente_mean":         "MNT_SLOPE_MEAN",
}

# ── Seuil de détection réglementaire (mg/kg) ─────────────────────────────────
DETECTION_THRESHOLD     = 0.1   # seuil bas (limite de détection)
REGULATORY_THRESHOLD    = 1.0   # seuil réglementaire restrictions agricoles
HIGH_CONTAMINATION      = 10.0  # contamination élevée

# ── Classes de contamination ──────────────────────────────────────────────────
CLASS_LABELS = {
    0: "Non détecté (<0.1)",
    1: "Faible (0.1–1.0)",
    2: "Modéré (1.0–10.0)",
    3: "Élevé (>10.0)",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes selon le dictionnaire des variables."""
    df = df.rename(columns=RENAME_MAP)
    logger.info(f"  🔤 Renommage : {len(RENAME_MAP)} colonnes renommées")
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes de dates en datetime."""
    date_cols = ["DATE_OF_SAMPLING", "DATE_RECORDED", "DATE_ANALYSIS"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    logger.info(f"  📅 Dates converties : {date_cols}")
    return df


def clean_rate_5b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie RATE_5B_HYDRO : valeurs comme '0,07' → 0.07, '<0.01' → 0.005, etc.
    """
    if "RATE_5B_HYDRO" not in df.columns:
        return df

    def parse_rate(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace(",", ".")
        if s.startswith("<"):
            try:
                return float(s[1:]) / 2  # Valeur de remplacement = moitié du seuil
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["RATE_5B_HYDRO"] = df["RATE_5B_HYDRO"].apply(parse_rate)
    logger.info("  🧪 RATE_5B_HYDRO nettoyé (virgules, < seuils)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables dérivées clés pour l'analyse et la modélisation.

    Variables créées :
    - DETECTED             : 1 si CLD > seuil de détection, 0 sinon
    - LOG_CHLD             : log(1 + CHLORDECONE_RATE) — transformation log-normale
    - CLASS_CONTAMINATION  : classe ordinale 0-3
    - ABOVE_REG_THRESHOLD  : booléen > seuil réglementaire 1.0 mg/kg
    - HISTOBANANE_HISTO_BAN: valeurs manquantes → 0 (pas de bananier historique)
    """
    # 1. Variable binaire de détection
    df["DETECTED"] = (df["CHLORDECONE_RATE"] > DETECTION_THRESHOLD).astype(int)
    logger.info(
        f"  🔍 DETECTED : {df['DETECTED'].sum():,} détections / {len(df):,} "
        f"({df['DETECTED'].mean()*100:.1f}%)"
    )

    # 2. Transformation logarithmique (gestion de la loi log-normale)
    df["LOG_CHLD"] = np.log1p(df["CHLORDECONE_RATE"])

    # 3. Classification en niveaux de contamination
    conditions = [
        df["CHLORDECONE_RATE"] <= DETECTION_THRESHOLD,
        (df["CHLORDECONE_RATE"] > DETECTION_THRESHOLD) & (df["CHLORDECONE_RATE"] <= REGULATORY_THRESHOLD),
        (df["CHLORDECONE_RATE"] > REGULATORY_THRESHOLD) & (df["CHLORDECONE_RATE"] <= HIGH_CONTAMINATION),
        df["CHLORDECONE_RATE"] > HIGH_CONTAMINATION,
    ]
    df["CLASS_CONTAMINATION"] = np.select(conditions, [0, 1, 2, 3], default=np.nan)
    logger.info("  📊 CLASS_CONTAMINATION créée (4 niveaux : 0–3)")

    # 4. Variable réglementaire
    df["ABOVE_REG_THRESHOLD"] = (df["CHLORDECONE_RATE"] > REGULATORY_THRESHOLD).astype(int)

    # 5. Imputation histobanane (valeurs manquantes = 0)
    if "HISTOBANANE_HISTO_BAN" in df.columns:
        n_missing = df["HISTOBANANE_HISTO_BAN"].isna().sum()
        df["HISTOBANANE_HISTO_BAN"] = df["HISTOBANANE_HISTO_BAN"].fillna(0)
        if n_missing:
            logger.info(f"  🍌 HISTOBANANE : {n_missing} valeurs manquantes → 0")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rapport et traitement minimal des valeurs manquantes.
    On conserve les lignes sauf si CHLORDECONE_RATE est manquant.
    """
    n_before = len(df)
    df = df.dropna(subset=["CHLORDECONE_RATE"])
    n_after = len(df)

    if n_before != n_after:
        logger.warning(
            f"  ⚠️  {n_before - n_after} lignes supprimées (CHLORDECONE_RATE manquant)"
        )

    # Rapport des valeurs manquantes restantes
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(f"  📋 Valeurs manquantes restantes :\n{missing.to_string()}")

    return df


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de nettoyage complet.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame brut issu de ingestion.load_raw()

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé et enrichi.
    """
    logger.info("🧹 Démarrage du nettoyage...")
    df = df_raw.copy()

    df = rename_columns(df)
    df = convert_dates(df)
    df = clean_rate_5b(df)
    df = handle_missing_values(df)
    df = engineer_features(df)

    logger.info(f"✅ Nettoyage terminé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


# ── Point d'entrée direct ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion import load_raw, validate_schema

    df_raw = load_raw()
    validate_schema(df_raw)
    df_clean = clean(df_raw)

    print("\n── Aperçu des nouvelles colonnes ──────────────────────")
    new_cols = ["DETECTED", "LOG_CHLD", "CLASS_CONTAMINATION", "ABOVE_REG_THRESHOLD"]
    print(df_clean[new_cols].describe())
    print(f"\nDistribution CLASS_CONTAMINATION :\n{df_clean['CLASS_CONTAMINATION'].value_counts().sort_index()}")
