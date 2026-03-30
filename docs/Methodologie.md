# 📐 Méthodologie — Analyse Contamination Chlordécone

## 1. Architecture du projet

Ce projet suit la structure standard d'un projet Data Science reproductible,
inspirée des bonnes pratiques enseignées en cours **Data Engineering (ENSAR 2026)** :

```
Données brutes → ETL (cleaning) → Analyse → Modélisation → Résultats
```

Chaque couche est **indépendante** et **importable** :
- On peut utiliser les modules `src/` depuis les notebooks ou en ligne de commande.
- Les données brutes ne sont jamais modifiées (principe d'immutabilité des données sources).

---

## 2. Choix du pipeline ETL

### Séparateur CSV
Le fichier source utilise `;` comme séparateur (convention française pour éviter
les conflits avec les virgules décimales). Détecté et documenté dans `ingestion.py`.

### Validation de schéma
Avant tout traitement, le schéma est validé (`validate_schema`) : toutes les colonnes
attendues (dictionnaire des 22 variables) doivent être présentes. Ce filet de sécurité
empêche des erreurs silencieuses si le fichier source change.

### Feature engineering — Transformation log
La transformation `LOG_CHLD = log(1 + Taux_CLD)` est fondamentale :
- Les polluants environnementaux suivent des distributions **log-normales** (processus multiplicatifs)
- La transformation stabilise la variance pour les modèles de régression
- Le `+1` évite `log(0)` pour les valeurs nulles (non détectés)

---

## 3. Choix des tests statistiques

### Pourquoi non-paramétrique ?
Les tests paramétriques (ANOVA, t-test de Student, Pearson) supposent :
1. La **normalité** des données ou des résidus
2. L'**homoscédasticité** (égalité des variances)

Le taux de chlordécone viole ces deux hypothèses (distribution log-normale,
hétéroscédasticité inter-groupes). On utilise donc systématiquement des tests
basés sur les **rangs**, insensibles à la distribution sous-jacente.

| Contexte | Test paramétrique | Test non-paramétrique choisi |
|---|---|---|
| 2 groupes indépendants | t-test de Student | Mann-Whitney U |
| k > 2 groupes indépendants | ANOVA one-way | Kruskal-Wallis H |
| 2 variables catégorielles | — | Chi-2 d'indépendance |
| Corrélation | Pearson r | Spearman ρ |

### Seuil α = 0.05
Seuil de significativité standard en sciences environnementales.
Le V de Cramér complète le χ² pour quantifier la **force** de l'association
(le χ² seul dépend de N et peut être significatif sans être pratiquement pertinent).

---

## 4. Choix de la modélisation — KNN

### Pourquoi KNN ?
- Algorithme enseigné en cours (Data Analysis — 27 Fév. 2026)
- Non-paramétrique : aucune hypothèse sur la distribution des données
- Interprétable : la prédiction s'explique par les voisins les plus proches
- Adaptable : fonctionne pour la classification et la régression

### Standardisation — Pourquoi obligatoire ?
KNN calcule des **distances euclidiennes** entre observations.
Sans standardisation, une variable avec de grandes valeurs numériques
(ex: `MNT_SHADING_MEAN` ≈ 130) dominerait la distance face à une variable
centrée sur 0 (ex: `MNT_TPI_MEAN` ≈ 5). Résultat : le modèle ignorerait de facto
toutes les variables à faible variance numérique.

La formule de standardisation Z-score : **x' = (x − μ) / σ**

**IMPORTANT** : le scaler est entraîné (`fit`) **uniquement sur le train**.
On applique ensuite (`transform`) aux deux sets. Inverser cet ordre constitue
un **data leakage** — le modèle "verrait" les données de test pendant l'entraînement.

### Validation croisée — 5-fold stratifié
Pour la classification, on utilise un split **stratifié** : chaque fold contient
la même proportion de classes 0/1 que le dataset complet.
Cela évite qu'un fold ne contienne que des non-détectés, rendant l'évaluation biaisée.

### Permutation Importance — Pourquoi pas les coefficients ?
KNN n'a pas de coefficients explicites (contrairement à la régression linéaire).
La **permutation importance** est la méthode *model-agnostic* standard :
on mesure la dégradation de performance quand on mélange une variable.
20 répétitions permettent d'estimer la variance de cette importance.

---

## 5. Cartographie — Système de référence

Les coordonnées X/Y du dataset sont en **EPSG:5490** (RGAF09 / UTM zone 20N),
système officiel pour les Antilles françaises.

Pour visualiser sur une carte web (Folium, OpenStreetMap) :
```python
gdf = gdf.to_crs(epsg=4326)   # Conversion vers WGS84 (lat/lon)
```

Les shapefiles fournis (Martinique, communes) sont au même CRS — aucune
reprojection n'est nécessaire pour les jointures spatiales.

---

## 6. Reproductibilité

- **`random_state=42`** : utilisé pour tous les splits et sous-échantillonnages
- **`n_jobs=-1`** : parallélisation sur tous les cœurs disponibles (sklearn)
- **`requirements.txt`** : toutes les versions de packages sont spécifiées
- **Données brutes immuables** : `data/raw/` jamais modifié, traitement dans `data/processed/`

---

## 7. Limitations et perspectives

### Limitations
- Le KNN ne capture pas la **dépendance spatiale** des données (krigeage préférable)
- Les variables topographiques (MNT) n'expliquent qu'une partie de la variance
- L'historique d'épandage précis par parcelle n'est pas disponible dans le dataset
- Le déséquilibre de classes (~56% de détections) peut biaiser la classification

### Perspectives
- **Krigeage ordinaire/universel** : méthode géostatistique qui modélise explicitement
  la structure spatiale de la contamination (variogramme)
- **Random Forest** : plus robuste que KNN, gère nativement les features catégorielles
- **Variables supplémentaires** : données cadastrales historiques, débit des cours d'eau,
  profondeur de sol
- **Modélisation temporelle** : séries temporelles par commune (SARIMA, Prophet)
