# Anomaly Detection - Transactions Ethereum

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![pandas](https://img.shields.io/badge/pandas-2.3-green)
![category_encoders](https://img.shields.io/badge/category__encoders-2.9-purple)

## Objectif

Détecter des adresses Ethereum frauduleuses à partir de leur comportement transactionnel, en utilisant une approche **non supervisée** (Isolation Forest).

## Dataset

- **Source** : `transaction_dataset.csv`
- **9 841 adresses** Ethereum, chacune décrite par ~50 features (fréquence des transactions, montants envoyés/reçus, diversité des contacts, activité ERC20...)
- **Variable cible** : `FLAG` (0 = légitime, 1 = frauduleuse) — utilisée uniquement pour l'évaluation, jamais pour l'entraînement
- **Déséquilibre** : ~78% légitimes, ~22% frauduleuses

## Démarche

### 1. Exploration (EDA)

- Nettoyage des noms de colonnes
- Vérification des doublons et suppression des colonnes identifiants (`Index`, `Address`)
- Analyse des valeurs manquantes et de leur distribution par rapport à `FLAG`
- Suppression des colonnes à variance nulle (7 colonnes constantes)
- Visualisation de la matrice de corrélation

### 2. Preprocessing

- **Split train/test (80/20)** avant toute transformation pour éviter le data leakage
- **Imputation** : médiane (numériques) et mode (catégorielles), calculés sur le train uniquement
- **Suppression des features corrélées** > 0.95 (7 colonnes redondantes)
- **Target Encoding** des 2 colonnes catégorielles (trop de modalités pour du one-hot)
- **RobustScaler** : normalisation via médiane/IQR, robuste aux valeurs extrêmes

### 3. Modélisation — Isolation Forest

L'Isolation Forest isole chaque point via des arbres de décision aléatoires. Les anomalies, étant éloignées de la majorité, nécessitent moins de coupures pour être isolées (chemin court = anomalie).

Le modèle s'entraîne **sans connaître `FLAG`**. Les labels servent uniquement à évaluer a posteriori.

### 4. Évaluation

| Contamination | F1 Légitime | F1 Fraude |
|---|---|---|
| `'auto'` (~3%) | 0.86 | 0.02 |
| `0.22` | 0.79 | 0.42 |
| `0.33` | 0.79 | 0.49 |

Le paramètre `contamination` fixe le seuil de décision (% de points classés comme anomalies). Il ne change pas l'entraînement, seulement la binarisation des scores.

### 5. Tuning des hyperparamètres

Grid search sur 27 combinaisons. Les hyperparamètres testés :

| Paramètre | Rôle | Défaut |
|---|---|---|
| `n_estimators` | Nombre d'arbres dans la forêt. Le score final est la moyenne de tous les arbres. Plus d'arbres = scores plus stables | 100 |
| `max_samples` | Nombre de points que chaque arbre reçoit pour s'entraîner (sous-échantillon aléatoire). Plus grand = meilleure vision de ce qui est "normal" | 256 |
| `max_features` | Proportion de features utilisées par chaque arbre. Réduire force la diversité entre arbres | 1.0 |
| `contamination` | Pourcentage de points classés comme anomalies. Ne change **pas** l'entraînement, fixe uniquement le **seuil** de décision sur le score | `'auto'` |

Les 3 premiers contrôlent **comment le modèle apprend**. Le dernier contrôle **comment il décide**.

Meilleure configuration retenue :

| Paramètre | Valeur |
|---|---|
| `n_estimators` | 100 |
| `max_samples` | 1024 |
| `max_features` | 1.0 |
| `contamination` | 0.22 |

**Résultat après tuning** : F1 légitime = **0.86**, F1 fraude = **0.49**

L'amélioration vient principalement de `max_samples=1024` (vs 256 par défaut) : chaque arbre voit 4x plus de données, ce qui affine sa représentation du comportement "normal".

### 6. Visualisation t-SNE

Projection des features en 2D pour vérifier visuellement la séparation entre points normaux et anomalies détectées.

## Non supervisé vs Supervisé : deux approches, deux usages

| | Non supervisé (ce projet) | Supervisé (projet SMOTE) |
|---|---|---|
| **Labels nécessaires ?** | Non (FLAG sert uniquement à évaluer) | Oui (le modèle apprend directement de FLAG) |
| **Ce que le modèle apprend** | La structure "normale" des données → détecte ce qui en dévie | La frontière entre classes 0 et 1 |
| **Gestion du déséquilibre** | Via `contamination` (seuil sur le score) | Via SMOTE (suréchantillonnage de la classe minoritaire) |
| **Performance typique** | Modérée (F1 fraude ~0.49 ici) | Meilleure (le modèle sait ce qu'est une fraude) |
| **Cas d'usage réel** | Pas de labels disponibles, exploration initiale, détection de nouveaux types de fraude | Labels disponibles, fraudes déjà connues et documentées |
| **Limite principale** | "Atypique" ≠ "frauduleux" (faux positifs, fraudes banales ratées) | Ne détecte que les fraudes qui ressemblent à celles déjà vues |

### En production

Le non supervisé est souvent utilisé en **première ligne** : il classe les adresses par score de suspicion et un analyste humain vérifie les plus suspectes. Au fil du temps, les vérifications humaines créent des labels qui permettent de passer progressivement au supervisé.

## Stack technique

- Python 3.13
- pandas, numpy, matplotlib, seaborn
- scikit-learn (Isolation Forest, t-SNE, RobustScaler)
- category_encoders (TargetEncoder)
