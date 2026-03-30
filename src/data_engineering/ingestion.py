"""
ingestion.py — Chargement et validation des données brutes
===========================================================
Cours mobilisé : Data Engineering (Séance 05 & 12 Janv., 26 Janv.)
Concepts : ETL (Extract), validation de schéma, typage, logging

Fonctions :
  - load_raw()       : charge le CSV brut avec validation
  - validate_schema(): vérifie présence et types des colonnes
  - quick_report()   : résumé rapide du dataset chargé
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Chemins ──────────────────────────────────────────────────────────────────
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "BaseCLD2026.csv"

# ── Schéma attendu (dictionnaire des variables) ───────────────────────────────
EXPECTED_COLUMNS = {
    "ID":                    "int64",
    "ANNEE":                 "int64",
    "COMMU_LAB":             "object",
    "RAIN":                  "object",
    "Sol_simple":            "object",
    "type_sol":              "object",
    "Date_prelevement":      "object",
    "Date_enregistrement":   "object",
    "Date_analyse":          "object",
    "Operateur_chld":        "object",
    "Taux_Chlordecone":      "float64",
    "Operateur_5b":          "object",
    "Taux_5b_hydro":         "object",
    "histoBanane_Histo_ban": "int64",
    "mnt_tpi_mean":          "float64",
    "mnt_tri_mean":          "float64",
    "mnt_rugosite_mean":     "float64",
    "mnt_ombrage_mean":      "float64",
    "mnt_exposition_mean":   "float64",
    "mnt_pente_mean":        "float64",
    "X":                     "float64",
    "Y":                     "float64",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Charge le fichier CSV brut (séparateur ';').

    Parameters
    ----------
    path : Path
        Chemin vers BaseCLD2026.csv

    Returns
    -------
    pd.DataFrame
        DataFrame brut non transformé.
    """
    logger.info(f"📂 Chargement depuis : {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Placez BaseCLD2026.csv dans data/raw/"
        )

    df = pd.read_csv(path, sep=";", low_memory=False)
    logger.info(f"✅ Chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Valide que toutes les colonnes attendues sont présentes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à valider.

    Returns
    -------
    dict
        Rapport de validation : colonnes manquantes, colonnes en plus.
    """
    actual_cols   = set(df.columns)
    expected_cols = set(EXPECTED_COLUMNS.keys())

    missing  = expected_cols - actual_cols
    extra    = actual_cols   - expected_cols

    report = {
        "nb_rows":         len(df),
        "nb_cols":         len(df.columns),
        "missing_columns": list(missing),
        "extra_columns":   list(extra),
        "is_valid":        len(missing) == 0,
    }

    if missing:
        logger.warning(f"⚠️  Colonnes manquantes : {missing}")
    else:
        logger.info("✅ Schéma valide — toutes les colonnes attendues sont présentes")

    if extra:
        logger.info(f"ℹ️  Colonnes supplémentaires (non attendues) : {extra}")

    return report


def quick_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport rapide : types, valeurs nulles, cardinalité.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Tableau récapitulatif par colonne.
    """
    report = pd.DataFrame({
        "Type":             df.dtypes,
        "Non-nuls":         df.notna().sum(),
        "Nuls":             df.isna().sum(),
        "% Nuls":           (df.isna().mean() * 100).round(2),
        "Valeurs uniques":  df.nunique(),
        "Exemple":          df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else "N/A"),
    })
    return report


# ── Point d'entrée direct ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = load_raw()
    rapport_schema = validate_schema(df_raw)
    print("\n── Rapport de schéma ──────────────────────────────────")
    for k, v in rapport_schema.items():
        print(f"  {k:20s}: {v}")

    print("\n── Rapport par colonne ────────────────────────────────")
    print(quick_report(df_raw).to_string())
