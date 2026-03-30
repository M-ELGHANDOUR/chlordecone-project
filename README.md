# 🌿 Analyse de la Contamination au Chlordécone — Martinique

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Contexte scientifique :** Le chlordécone est un pesticide organochloré utilisé massivement dans les plantations de bananes aux Antilles françaises de 1972 à 1993. En raison de sa très haute persistance chimique (demi-vie estimée à plusieurs centaines d'années dans les sols argileux tropicaux), il constitue aujourd'hui l'une des pollutions agricoles les plus durables connues. Ce projet analyse **31 126 mesures** de contamination des sols en Martinique (2010–2019).

---

## 📁 Structure du Projet

```
chlordecone-project/
│
├── data/
│   ├── raw/                   # Données brutes originales (ne pas modifier)
│   │   └── BaseCLD2026.csv
│   ├── processed/             # Données nettoyées et enrichies
│   │   └── chlordecone_clean.csv
│   └── external/              # Shapefiles Martinique (cartographie)
│       ├── Martinique.shp / .dbf / .prj / .shx
│       └── ARRONDISSEMENT.shp / ...
│
├── notebooks/
│   ├── 01_data_engineering.ipynb     # Ingestion, nettoyage, pipeline
│   ├── 02_data_analysis.ipynb        # EDA, stats, visualisations
│   └── 03_modeling.ipynb             # KNN classification & régression
│
├── src/
│   ├── data_engineering/
│   │   ├── ingestion.py              # Chargement et validation des données
│   │   ├── cleaning.py               # Nettoyage et transformation
│   │   └── pipeline.py               # Pipeline ETL complet
│   ├── data_analysis/
│   │   ├── stats.py                  # Tests statistiques
│   │   ├── visualization.py          # Graphiques et cartes
│   │   └── spatial.py                # Analyse spatiale GeoPandas
│   └── modeling/
│       ├── preprocessing.py          # Feature engineering pour ML
│       ├── knn_classifier.py         # KNN Classification
│       └── knn_regressor.py          # KNN Régression
│
├── reports/
│   └── figures/                      # Graphiques exportés
│
├── docs/
│   ├── Dictionnaire_Variables.md     # Description de toutes les variables
│   └── Methodologie.md               # Choix méthodologiques
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Démarrage

### 1. Cloner le dépôt
```bash
git clone https://github.com/TON_USERNAME/chlordecone-martinique.git
cd chlordecone-martinique
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer le pipeline ETL
```bash
python src/data_engineering/pipeline.py
```

### 5. Explorer les notebooks
```bash
jupyter lab
```

---

## 🔬 Pipeline d'Analyse

```
BaseCLD2026.csv (raw)
       │
       ▼
 [Data Engineering]
  • Validation du schéma
  • Nettoyage (types, encodage)
  • Feature engineering
  • Export → processed/
       │
       ▼
 [Data Analysis]
  • Statistiques descriptives
  • Tests non-paramétriques (Kruskal-Wallis, Mann-Whitney, χ²)
  • Corrélations (Pearson & Spearman)
  • Analyse spatiale (GeoPandas)
  • Analyse temporelle (2010–2019)
       │
       ▼
 [Modeling — KNN]
  • Classification : détecté / non détecté
  • Régression : prédiction du taux (log-transformé)
  • Validation croisée 5-fold
  • Permutation importance
```

---

## 📊 Résultats Clés

| Indicateur | Valeur |
|---|---|
| Observations totales | 31 126 |
| Communes couvertes | 35 |
| Période | 2010 – 2019 |
| Taux de détection | ~56% |
| Taux médian (détectés) | ~0.5 mg/kg |
| Meilleur K (KNN clf) | 11 |
| F1-score KNN (validation) | ~0.77 |

---

## 📚 Cours et Concepts Mobilisés

| Module | Concepts |
|---|---|
| **Data Engineering** | ETL, pipelines, Git, Pandas, validation de schéma |
| **Data Governance** | Dictionnaire de données, traçabilité, qualité |
| **Data Analysis** | Statistiques descriptives, tests non-paramétriques, corrélations |
| **Cartographie** | GeoPandas, shapefiles, projections EPSG:5490 |
| **Cloud / Versionning** | Git, GitHub, structure de projet reproductible |
| **Data Science / ML** | KNN, validation croisée, feature engineering, permutation importance |

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre des cours **Data Engineering & Data Analysis — ENSAR 2026**

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)
