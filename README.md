# Analyse des Réseaux Sociaux Facebook100

Lancelot Dallain  
Master TRIED  
Janvier 2026

## Description

Analyse empirique de 100 réseaux sociaux universitaires du jeu de données Facebook100 (septembre 2005). 

Le projet examine cinq aspects fondamentaux :

- Structure des réseaux : distributions de degrés, coefficients de clustering, corrélations
- Assortativité : homophilie à travers 5 attributs (résidence, degré, genre, statut, spécialité)
- Prédiction de liens : métriques classiques (Voisins Communs, Jaccard, Adamic-Adar)
- Propagation de labels : inférence d'attributs manquants par apprentissage semi-supervisé
- Détection de communautés : optimisation gloutonne de modularité

## Résultats principaux

- Forte homophilie géographique : r=0,175 pour résidences
- 48% de précision en prédiction de liens avec voisins communs
- 100% de précision en propagation de labels pour l'attribut spécialité
- Structure de petit monde confirmée sur tous les réseaux

## Structure du projet

```
EXAM_FINAL/
├── q2_analysis.py              # Analyse structurelle
├── q3_assortativity.py         # Assortativité sur 100 réseaux
├── q4_link_prediction.py       # Prédiction de liens
├── q5_label_propagation.py     # Propagation de labels
├── q6_community.py             # Détection de communautés
├── homework_report.tex         # Rapport LaTeX complet
├── lib/                        # Bibliothèques réutilisables
└── results/                    # Figures et résultats
    ├── q2/
    ├── q3/
    ├── q4/
    ├── q5/
    └── q6/
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Analyse complète
python main.py

# Analyses individuelles
python q2_analysis.py
python q3_assortativity.py
```

## Données

Jeu de données Facebook100 : 100 réseaux sociaux universitaires (septembre 2005)

- 1,2 millions de nœuds
- 93 millions d'arêtes
- Attributs : statut, spécialité, résidence, genre

## Dépendances

Python 3.12+, NetworkX, NumPy, pandas, matplotlib, seaborn

## Rapport

Le rapport complet est disponible dans `homework_report.tex`.
