# 🌿 Analyse de la Contamination au Chlordécone — Martinique

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Folium](https://img.shields.io/badge/Folium-Maps-77B829?style=for-the-badge)](https://python-visualization.github.io/folium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 👤 À propos de moi

Je suis **Mohsine Elghandour**, étudiant en **Data Science à l'ENSAR Poitiers**
(École Nationale Supérieure d'Agronomie et des industries alimentaires),
**actuellement à la recherche d'une alternance de 2026 à 2028** dans les domaines
de la Data Science, du Machine Learning ou de l'IA appliquée.

Ce projet est réalisé dans le cadre de mes cours. Il représente mon travail concret
sur un vrai jeu de données environnemental — du nettoyage brut jusqu'à la modélisation
et la cartographie interactive.

> 📬 N'hésitez pas à me contacter si vous cherchez un alternant motivé et rigoureux !
>
> +33759459928
> mohsine.elghandour@etu.univ-poitiers.fr

---

## 🧭 Contexte du Projet — Pourquoi ce sujet ?

Le **chlordécone** est un pesticide organochloré qui a été épandu massivement dans les
plantations de bananes aux Antilles françaises (**Martinique et Guadeloupe**) de
**1972 à 1993**. Il a été utilisé pour protéger les cultures du charançon du bananier.

Le problème : ce produit est **extrêmement persistant dans les sols**.
Sa demi-vie est estimée à **plusieurs centaines d'années** dans les sols argileux tropicaux.
Cela signifie qu'aujourd'hui, plus de **30 ans après son interdiction**, les terres agricoles
de Martinique sont toujours contaminées.

Les conséquences sont graves :
- Restrictions sur la production agricole locale (maraîchage, élevage)
- Contamination des eaux et des produits de la mer
- Exposition chronique des populations — le chlordécone est classé **cancérogène possible**
  (CIRC, groupe 2B) et est lié à une surincidence du cancer de la prostate aux Antilles

**Ce projet analyse 31 126 mesures** de contamination des sols en Martinique entre
**2010 et 2019**, pour comprendre comment et où la contamination se distribue,
quels facteurs l'influencent, et comment un modèle de Machine Learning peut prédire
le niveau de contamination d'une parcelle.

---

## 📁 Structure du Projet

```
chlordecone-project/
│
├── 📂 data/
│   ├── raw/                               # Données brutes — ne jamais modifier
│   │   └── BaseCLD2026.csv                # 31 126 mesures de sols (2010–2019)
│   ├── processed/                         # Données nettoyées et enrichies
│   │   └── chlordecone_clean.csv
│   └── external/                          # Shapefiles Martinique (cartographie)
│       ├── Martinique.shp / .dbf / .prj / .shx
│       └── ARRONDISSEMENT.shp / ...
│
├── 📓 notebooks/
│   ├── 01_data_engineering.ipynb          # Étape 1 : Ingestion, validation, nettoyage
│   ├── 02_data_analysis.ipynb             # Étape 2 : Statistiques, tests, visualisations
│   ├── 03_modeling.ipynb                  # Étape 3 : Modélisation KNN
│   └── 04_cartographie_interactive.ipynb  # Étape 4 : Cartes HTML interactives
│
├── 📦 src/
│   ├── data_engineering/
│   │   ├── ingestion.py                   # Chargement et validation du schéma
│   │   ├── cleaning.py                    # Nettoyage, feature engineering
│   │   └── pipeline.py                    # Pipeline ETL complet (CLI)
│   ├── data_analysis/
│   │   ├── stats.py                       # Tests statistiques (KS, KW, MW, χ², Spearman)
│   │   ├── visualization.py               # Graphiques matplotlib/seaborn
│   │   └── spatial.py                     # Analyse spatiale GeoPandas
│   └── modeling/
│       ├── preprocessing.py               # Feature engineering pour ML
│       ├── knn_classifier.py              # KNN Classification + permutation importance
│       └── knn_regressor.py               # KNN Régression + courbe biais-variance
│
├── 📊 reports/
│   ├── figures/                           # Graphiques exportés (.png)
│   ├── carte1_points_contamination.html   # Points colorés + popup + légende
│   ├── carte2_heatmap.html                # Heatmap densité
│   ├── carte3_communes_bubble.html        # Bubble map par commune
│   └── carte4_multicouches.html           # Multi-couches avec LayerControl
│
├── 📄 docs/
│   ├── Dictionnaire_Variables.md          # Description détaillée de chaque variable
│   └── Methodologie.md                    # Justification des choix méthodologiques
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Démarrage rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/MohsineElghandour/chlordecone-martinique.git
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

### 4. Placer les données brutes
```
Copier BaseCLD2026.csv dans data/raw/
```

### 5. Lancer le pipeline ETL (nettoyage automatique)
```bash
python src/data_engineering/pipeline.py
```

### 6. Ouvrir les notebooks dans l'ordre
```bash
jupyter lab
# → 01_data_engineering.ipynb
# → 02_data_analysis.ipynb
# → 03_modeling.ipynb
# → 04_cartographie_interactive.ipynb
```

---

## 🔬 Ce que j'ai fait — Étape par étape

### 🔧 Étape 1 — Data Engineering : Construire un pipeline propre

Avant de toucher à n'importe quelle statistique ou modèle, j'ai commencé par
**comprendre et structurer les données brutes**.

Le fichier `BaseCLD2026.csv` contient 31 126 lignes et 22 colonnes avec des données
hétérogènes : des dates en texte, des taux avec des virgules au lieu de points, des
colonnes avec des valeurs `<0.01` (limites de détection), et des coordonnées géographiques
en projection UTM (pas en latitude/longitude standard).

**Ce que j'ai fait concrètement :**

- ✅ **Validation du schéma** : j'ai écrit une fonction qui vérifie que toutes les colonnes
  attendues sont présentes avant de lancer le pipeline. Si une colonne manque, le pipeline
  s'arrête proprement avec un message d'erreur clair — au lieu de planter silencieusement
  plus tard.

- ✅ **Renommage cohérent** : les noms de colonnes originaux étaient en français et en format
  incohérent (`mnt_tpi_mean`, `histoBanane_Histo_ban`…). Je les ai tous renommés en anglais,
  en snake_case, selon le dictionnaire des variables fourni avec le dataset.

- ✅ **Nettoyage du taux 5β-hydro** : la colonne `Taux_5b_hydro` contenait des valeurs
  textuelles comme `"<0.01"` ou `"0,07"` (virgule française). J'ai écrit un parseur qui
  convertit `<0.01` → `0.005` (moitié du seuil, convention statistique) et remplace les
  virgules par des points.

- ✅ **Feature engineering** : j'ai créé 4 nouvelles variables clés pour l'analyse
  et la modélisation :
  - `DETECTED` : 1 si le taux dépasse 0.1 mg/kg (seuil de détection analytique)
  - `LOG_CHLD` : log(1 + taux) — transformation logarithmique pour normaliser la distribution
  - `CLASS_CONTAMINATION` : classification en 4 niveaux (0 = propre → 3 = très contaminé)
  - `ABOVE_REG_THRESHOLD` : 1 si le taux dépasse le seuil réglementaire de 1.0 mg/kg

- ✅ **Pipeline ETL en ligne de commande** : tout le processus Extract → Transform → Load
  est encapsulé dans `pipeline.py`, exécutable en une seule commande avec des paramètres
  configurables via argparse.

---

### 📊 Étape 2 — Data Analysis : Comprendre les données en profondeur

Une fois les données propres, j'ai mené une analyse statistique complète pour répondre
à des questions concrètes.

**❓ Question 1 : Comment se distribue le taux de chlordécone ?**

J'ai tracé la distribution brute : elle est fortement **asymétrique à droite** (skewness élevé).
La plupart des parcelles ont un taux faible, mais quelques-unes ont des valeurs très élevées
qui tirent la moyenne vers le haut. C'est typique des polluants environnementaux — ils suivent
une **loi log-normale** (résultat de processus multiplicatifs : dilution, dégradation,
accumulation).

J'ai vérifié cette hypothèse avec deux tests statistiques :
- **Test de Kolmogorov-Smirnov** : compare la distribution empirique à une loi normale théorique
- **Test de Shapiro-Wilk** : test de normalité exact (sur sous-échantillon < 5000 obs.)

Résultat : après transformation logarithmique, la distribution s'approche d'une loi normale ✅

**❓ Question 2 : Le type de sol influence-t-il la contamination ?**

Les Andosols (sols volcaniques riches en allophane) sont connus pour fixer fortement
les molécules organochlorées. J'ai testé si les différences entre types de sol étaient
statistiquement significatives avec le **test de Kruskal-Wallis** — équivalent non-paramétrique
de l'ANOVA, utilisé ici car la distribution n'est pas normale.

Résultat : **p << 0.05** — le type de sol a un effet hautement significatif sur la contamination.
Le **test de Mann-Whitney** en comparaison deux à deux confirme que les différences entre sols
spécifiques sont réelles et non dues au hasard.

**❓ Question 3 : Les variables topographiques sont-elles liées à la contamination ?**

J'ai calculé les corrélations de **Spearman** (plutôt que Pearson, car les données sont
asymétriques et non-normales) entre le taux CLD et les variables de relief : pente, TPI,
exposition, rugosité, ombrage.

Un **test du Chi-2** complète l'analyse pour mesurer l'association entre le type de sol
et la classe de contamination, avec le **V de Cramér** pour quantifier la force de cette
association (le χ² seul dépend de N et peut être significatif sans être pratiquement pertinent).

**❓ Question 4 : La contamination évolue-t-elle dans le temps ?**

J'ai tracé l'évolution du taux médian par année entre 2010 et 2019.
Résultat frappant : **aucune tendance à la baisse** sur 10 ans. Malgré 30 ans sans épandage,
les taux restent stables. Cela confirme scientifiquement la persistance exceptionnelle
du chlordécone dans les sols.

**❓ Question 5 : Quelles communes sont les plus touchées ?**

Classement des communes par taux médian et par pourcentage de dépassement du seuil
réglementaire — avec visualisation en bar chart horizontal et mise en évidence des
communes dépassant le seuil de 1 mg/kg.

---

### 🤖 Étape 3 — Modélisation KNN : Prédire la contamination

J'ai appliqué l'algorithme **K-Nearest Neighbors (KNN)** pour deux tâches différentes.

**Pourquoi KNN ?**
C'est un algorithme non-paramétrique (aucune hypothèse sur la distribution des données),
intuitif à comprendre (la prédiction s'explique par les voisins les plus proches),
et il fonctionne pour la classification comme pour la régression.

**Features utilisées :**
Les variables topographiques (pente, TPI, TRI, exposition, ombrage, rugosité),
le type de sol encodé numériquement, et le nombre de périodes historiques de bananier —
proxy de l'exposition passée au pesticide.

**Preprocessing — Standardisation obligatoire :**
KNN est basé sur des distances euclidiennes. Sans standardisation, une variable
avec de grandes valeurs (ex : ombrage ≈ 130) dominerait toutes les distances face
à une variable proche de 0. J'ai appliqué un `StandardScaler` entraîné **uniquement
sur le train** pour éviter tout data leakage vers le jeu de test.

**Tâche 1 — Classification binaire : cette parcelle est-elle contaminée ?**
- Cible : `DETECTED` (0 = non détecté, 1 = détecté)
- K optimal sélectionné par **validation croisée 5-fold** sur le F1-score
- Évaluation : matrice de confusion, précision, rappel, F1-score
- Point clé : dans un contexte sanitaire, les **faux négatifs** (parcelle contaminée
  classée propre) sont l'erreur la plus dangereuse — je surveille le rappel autant que
  la précision.

**Tâche 2 — Régression : quel est le taux exact ?**
- Cible : `LOG_CHLD` (log du taux — transformé pour stabiliser la variance)
- K optimal en minimisant le **RMSE** par validation croisée
- Visualisation de la **courbe biais-variance** : K petit = surapprentissage, K grand =
  sous-apprentissage

**Permutation Importance :**
Pour comprendre quelles variables comptent le plus, j'ai utilisé la **permutation importance** :
on mélange aléatoirement les valeurs d'une variable et on mesure combien cela dégrade
les performances. Les variables historiques (bananier) et pédologiques (type de sol)
ressortent comme les plus influentes.

---

### 🗺️ Étape 4 — Cartographie Interactive : Voir les données sur la carte

J'ai créé 4 cartes HTML interactives avec **Folium** (basé sur Leaflet.js),
consultables directement dans un navigateur sans aucune installation.

**Bug corrigé par rapport à la carte initiale :**
Les coordonnées `X/Y` du dataset sont en **EPSG:5490** (projection UTM zone 20N, unité = mètres),
mais Folium attend du **WGS84** (latitude/longitude en degrés). Sans correction, les points
étaient placés à des coordonnées absurdes (~1 600 000, ~700 000) au lieu de (~14.7°N, ~61°W).
J'ai ajouté une reprojection avec `pyproj` pour corriger ça.

| Carte | Ce qu'elle montre |
|---|---|
| 🗺️ **Carte 1** | Points colorés vert/orange/rouge, **popup au clic** (commune, taux, sol, année) et légende fixe |
| 🔥 **Carte 2** | **Heatmap** de densité pondérée par log(CLD), fond sombre, gradient vert→rouge |
| 🫧 **Carte 3** | **Bubble map par commune** — taille = % dépassement réglementaire, couleur = taux médian |
| 🔀 **Carte 4** | **Multi-couches** avec LayerControl (on/off par niveau) + 3 fonds de carte sélectionnables |

---

## 📊 Résultats Clés

| Indicateur | Valeur |
|---|---|
| Observations totales | **31 126** |
| Communes couvertes | **35** |
| Période | **2010 – 2019** |
| Taux de détection réel | **34.6%** |
| % dépassant le seuil réglementaire (1 mg/kg) | **~8%** |
| Test Kruskal-Wallis (sol × CLD) | **p << 0.001** — hautement significatif |
| Évolution temporelle 2010–2019 | **Aucune baisse** — contamination stable |
| K optimal KNN — Classification | Sélectionné par CV 5-fold sur F1-score |
| K optimal KNN — Régression | Sélectionné par CV 5-fold sur RMSE |

---

## 📚 Cours et Compétences Mobilisés

| Module de cours | Concepts appliqués dans ce projet |
|---|---|
| **Data Engineering** | Pipeline ETL, validation de schéma, feature engineering, logging, CLI argparse |
| **Data Governance** | Dictionnaire des variables, traçabilité des transformations, principe d'immuabilité des données brutes |
| **Data Analysis** | Statistiques descriptives, loi log-normale, tests non-paramétriques (KW, MW, χ², KS, Shapiro-Wilk), corrélations Pearson & Spearman |
| **Cartographie Python** | GeoPandas, shapefiles, projections EPSG:5490, Folium, Leaflet, reprojection pyproj |
| **Git & Versionning** | Structure de projet reproductible, .gitignore, séparation raw/processed |
| **Data Science / ML** | KNN classification & régression, StandardScaler, train/test split stratifié, validation croisée 5-fold, permutation importance, courbe biais-variance |

---

## 🛠️ Stack Technique

```
Python 3.10+
├── pandas / numpy          → manipulation des données
├── scipy                   → tests statistiques
├── matplotlib / seaborn    → visualisations statiques
├── scikit-learn            → KNN, preprocessing, validation croisée
├── folium / pyproj         → cartographie interactive, reprojection
├── geopandas / shapely     → analyse spatiale (optionnel)
└── jupyter lab             → notebooks interactifs
```

---

## ⚠️ Limites et Perspectives

**Limites actuelles :**
- Le KNN ne capture pas la **dépendance spatiale** des données (deux parcelles géographiquement
  proches tendent à avoir des taux similaires — autocorrélation spatiale non modélisée)
- Les variables topographiques seules n'expliquent qu'une partie de la variance — l'historique
  précis d'épandage par parcelle n'est pas disponible dans le dataset
- Le déséquilibre de classes (34.6% de détections) peut biaiser certaines métriques

**Pistes d'amélioration :**
- **Krigeage ordinaire** : méthode géostatistique qui modélise explicitement la structure
  spatiale via le variogramme — idéale pour l'interpolation de polluants
- **Random Forest / XGBoost** : plus robustes que KNN, gèrent nativement les features
  catégorielles et les valeurs manquantes
- **Variables supplémentaires** : données cadastrales historiques d'épandage, profondeur
  de sol, proximité aux cours d'eau

---

## 👨‍💻 Auteur

**Mohsine Elghandour**
Étudiant Data Scientist — ENSAR Poitiers
🔍 *En recherche d'alternance Data Science / ML / IA — 2026 à 2028*

> Ce projet illustre ma capacité à mener un projet Data Science complet de bout en bout :
> compréhension d'un problème réel, nettoyage des données, analyse statistique rigoureuse,
> modélisation Machine Learning et communication des résultats via des visualisations interactives.

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)
