"""
pipeline.py — Pipeline ETL complet (Extract → Transform → Load)
================================================================
Cours mobilisé : Data Engineering (Séances Janv.–Mars 2026)
Concepts : Pipeline ETL, reproductibilité, logging structuré, export

Usage :
  python src/data_engineering/pipeline.py
  python src/data_engineering/pipeline.py --input data/raw/BaseCLD2026.csv
"""

import argparse
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
import pandas as pd

# Ajouter src/ au PYTHONPATH si exécuté directement
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engineering.ingestion import load_raw, validate_schema, quick_report
from data_engineering.cleaning  import clean


# ── Chemins par défaut ────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
RAW_PATH     = ROOT / "data" / "raw"     / "BaseCLD2026.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "chlordecone_clean.csv"


def run_pipeline(
    input_path: Path  = RAW_PATH,
    output_path: Path = PROCESSED_PATH,
    verbose: bool     = True,
) -> pd.DataFrame:
    """
    Exécute le pipeline ETL complet.

    Parameters
    ----------
    input_path  : chemin vers le CSV brut
    output_path : chemin de sortie du CSV nettoyé
    verbose     : afficher les rapports détaillés

    Returns
    -------
    pd.DataFrame — données nettoyées
    """
    logger.info("=" * 60)
    logger.info("  🚀 PIPELINE CHLORDÉCONE — Démarrage ETL")
    logger.info("=" * 60)

    # ── EXTRACT ───────────────────────────────────────────────────────────────
    logger.info("\n📥 ÉTAPE 1 : EXTRACT")
    df_raw = load_raw(input_path)

    schema_report = validate_schema(df_raw)
    if not schema_report["is_valid"]:
        logger.error(f"Schéma invalide : {schema_report['missing_columns']}")
        raise ValueError("Arrêt du pipeline — colonnes manquantes.")

    if verbose:
        logger.info("\n── Rapport initial ───────────────────────────────────")
        print(quick_report(df_raw).to_string())

    # ── TRANSFORM ─────────────────────────────────────────────────────────────
    logger.info("\n🔧 ÉTAPE 2 : TRANSFORM")
    df_clean = clean(df_raw)

    # Rapport post-nettoyage
    logger.info(f"\n  Nouvelles colonnes créées :")
    new_cols = [c for c in df_clean.columns if c not in df_raw.columns]
    for col in new_cols:
        logger.info(f"    + {col}")

    # ── LOAD ──────────────────────────────────────────────────────────────────
    logger.info("\n💾 ÉTAPE 3 : LOAD")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, sep=";")
    logger.info(f"✅ Données exportées → {output_path}")
    logger.info(f"   {df_clean.shape[0]:,} lignes × {df_clean.shape[1]} colonnes")

    # ── RÉSUMÉ ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  📋 RÉSUMÉ DU PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Entrée  : {input_path.name} ({df_raw.shape[0]:,} obs.)")
    logger.info(f"  Sortie  : {output_path.name} ({df_clean.shape[0]:,} obs.)")
    n_detected = df_clean['DETECTED'].sum() if 'DETECTED' in df_clean.columns else "N/A"
    pct = df_clean['DETECTED'].mean()*100 if 'DETECTED' in df_clean.columns else 0
    logger.info(f"  Détections CLD : {n_detected:,} ({pct:.1f}%)")
    logger.info(f"  Nouvelles variables : {len(new_cols)}")

    return df_clean


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline ETL — Contamination Chlordécone Martinique"
    )
    parser.add_argument(
        "--input", type=Path, default=RAW_PATH,
        help="Chemin vers le CSV brut"
    )
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_PATH,
        help="Chemin de sortie du CSV nettoyé"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Désactiver les rapports détaillés"
    )
    args = parser.parse_args()

    df = run_pipeline(
        input_path  = args.input,
        output_path = args.output,
        verbose     = not args.quiet,
    )
