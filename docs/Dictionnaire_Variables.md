# 📖 Dictionnaire des Variables — Contamination Chlordécone Martinique

Source : Dictionnaire des données de parcelles agricoles et mesures de contamination au chlordécone (INRAE/CIRAD).

---

## Variables du dataset brut (`BaseCLD2026.csv`)

| Variable originale | Variable renommée | Type | Description |
|---|---|---|---|
| `ID` | `ID` | int | Identifiant unique de la parcelle agricole |
| `ANNEE` | `YEAR` | int | Année d'observation ou de prélèvement (2010–2019) |
| `COMMU_LAB` | `COMMU_LAB` | str | Nom de la commune où se situe la parcelle |
| `RAIN` | `RAIN` | str | Groupe de pluviométrie (ex: 2000-3000 mm/an) |
| `Sol_simple` | `SOL_SIMPLE` | str | Type de sol simplifié (Andosol, Nitisol, Ferralsol…) |
| `type_sol` | `TYPE_SOL` | str | Type de sol détaillé (classification pédologique complète) |
| `Date_prelevement` | `DATE_OF_SAMPLING` | date | Date de prélèvement du sol sur la parcelle |
| `Date_enregistrement` | `DATE_RECORDED` | date | Date d'enregistrement au laboratoire |
| `Date_analyse` | `DATE_ANALYSIS` | date | Date d'analyse en laboratoire |
| `Operateur_chld` | `OPERATOR_CLD` | str | Opérateur de la quantification du chlordécone |
| `Taux_Chlordecone` | `CHLORDECONE_RATE` | float | Taux mesuré de chlordécone (mg/kg de sol sec) |
| `Operateur_5b` | `OPERATOR_5B` | str | Opérateur de la quantification du 5β-hydro |
| `Taux_5b_hydro` | `RATE_5B_HYDRO` | float | Taux de 5β-hydro chlordécone (métabolite, mg/kg) |
| `histoBanane_Histo_ban` | `HISTOBANANE_HISTO_BAN` | int | Nombre de périodes historiques de culture bananière |
| `mnt_tpi_mean` | `MNT_TPI_MEAN` | float | TPI moyen (Topographic Position Index) — position dans le relief |
| `mnt_tri_mean` | `MNT_TRI_MEAN` | float | TRI moyen (Terrain Ruggedness Index) — rugosité terrain |
| `mnt_rugosite_mean` | `MNT_RUGOSITY_MEAN` | float | Rugosité moyenne du terrain |
| `mnt_ombrage_mean` | `MNT_SHADING_MEAN` | float | Ombrage moyen de la parcelle (°) |
| `mnt_exposition_mean` | `MNT_EXPOSURE_MEAN` | float | Exposition moyenne du terrain (orientation, °) |
| `mnt_pente_mean` | `MNT_SLOPE_MEAN` | float | Pente moyenne du terrain (°) |
| `X` | `X` | float | Coordonnée X — projection EPSG:5490 (RGAF09 / UTM 20N) |
| `Y` | `Y` | float | Coordonnée Y — projection EPSG:5490 (RGAF09 / UTM 20N) |

---

## Variables dérivées (feature engineering)

| Variable | Type | Définition | Usage |
|---|---|---|---|
| `DETECTED` | int (0/1) | 1 si `CHLORDECONE_RATE` > 0.1 mg/kg | Cible classification KNN |
| `LOG_CHLD` | float | log(1 + `CHLORDECONE_RATE`) | Cible régression KNN, normalise la distribution |
| `CLASS_CONTAMINATION` | int (0-3) | Classe de contamination (voir ci-dessous) | Cible classification multi-classe |
| `ABOVE_REG_THRESHOLD` | int (0/1) | 1 si `CHLORDECONE_RATE` > 1.0 mg/kg | Indicateur réglementaire |
| `SOL_SIMPLE_ENCODED` | int | Label encoding du type de sol | Feature ML |

### Classes de contamination (`CLASS_CONTAMINATION`)

| Classe | Intervalle | Signification |
|---|---|---|
| 0 | CLD ≤ 0.1 mg/kg | Non détecté — en dessous du seuil de détection analytique |
| 1 | 0.1 < CLD ≤ 1.0 mg/kg | Faible — détecté mais sous le seuil réglementaire |
| 2 | 1.0 < CLD ≤ 10.0 mg/kg | Modéré — au-dessus du seuil : restrictions agricoles possibles |
| 3 | CLD > 10.0 mg/kg | Élevé — contamination sévère : interdictions d'usage du sol |

---

## Seuils réglementaires de référence

| Seuil | Valeur | Signification |
|---|---|---|
| Limite de détection | 0.1 mg/kg | En-dessous : valeur non quantifiable analytiquement |
| Seuil réglementaire | 1.0 mg/kg | Au-dessus : restrictions agricoles (arrêté préfectoral) |
| Contamination élevée | 10.0 mg/kg | Zones à risque élevé, interdiction de production maraîchère |

---

## Contexte scientifique

Le **chlordécone** (C₁₀Cl₁₀O) est un pesticide organochloré polycyclique.
Utilisé en Martinique et Guadeloupe de **1972 à 1993** pour lutter contre le charançon
du bananier (*Cosmopolites sordidus*), il est classé **cancérogène possible** (CIRC, groupe 2B).

Sa **demi-vie** dans les sols argileux tropicaux est estimée à **plusieurs centaines d'années**
(certaines études évoquent 600 ans), ce qui en fait l'une des pollutions agricoles
les plus durables connues.

### Variables topographiques MNT

Les variables préfixées `MNT_` sont dérivées d'un **Modèle Numérique de Terrain (MNT)**
à haute résolution. Elles caractérisent la topographie de chaque parcelle :

- **TPI** (*Topographic Position Index*) : différence d'altitude entre le point et la moyenne locale.
  Valeur positive = sommet/crête ; valeur négative = vallée/dépression.
  Les zones dépressionnaires accumulent le ruissellement → accumulation potentielle de CLD.

- **TRI** (*Terrain Ruggedness Index*) : hétérogénéité du relief local.
  Calculé comme l'écart-type des altitudes dans un rayon donné.

- **Rugosité** : irrégularité de surface du terrain.

- **Pente** : angle d'inclinaison du terrain (°). Influence l'érosion et le lessivage.

- **Exposition** : orientation du versant par rapport au nord (°).
  Influence l'ensoleillement, l'évapotranspiration et la dégradation photolytique.

- **Ombrage** : proportion du temps ombragé (influence la température du sol).

---

## Système de coordonnées

- **EPSG:5490** — RGAF09 / UTM zone 20N
- Système officiel pour les Antilles françaises (Martinique, Guadeloupe)
- Unités : mètres
- À convertir en EPSG:4326 (WGS84, lat/lon) pour les cartes Folium/web
