# Baseline Z-LIN CIFAR-10 — Design

## Objectif

Aligner une expérience séparée du projet sur la baseline Z-LIN de
« How far can we go without convolution », sans augmentation spatiale, afin
de rendre la comparaison avec le résultat publié reproductible et explicite.

Le MLP pédagogique actuel reste conservé comme expérience distincte.

## Périmètre

La baseline comprendra :

- CIFAR-10 standard : 50 000 images d'entraînement et 10 000 images de test ;
- centrage, PCA et blanchiment ajustés uniquement sur l'entraînement ;
- architecture Z-LIN :
  `4000Z -> 1000Lin -> 4000Z -> 1000Lin -> 4000Z -> 1000Lin -> 4000Z -> 10` ;
- dropout conforme au protocole de référence, avec taux de fine-tuning de 0,1 ;
- momentum SGD à 0,9 ;
- taux d'apprentissage distincts pour le pré-entraînement et le fine-tuning ;
- pré-entraînement couche par couche avec autoencodeurs ;
- fine-tuning supervisé de l'ensemble du réseau ;
- évaluation séparée sur validation et test, sans utiliser le test pour les décisions.

L'augmentation d'images ne sera pas activée dans cette baseline. Elle fera
l'objet d'une expérience ultérieure et explicitement identifiée.

## Architecture logicielle

### Prétraitement

Créer un transformateur PCA/whitening ayant un cycle explicite :

1. `fit(X_train)` calcule moyenne, composantes et échelles ;
2. `transform(X)` applique les paramètres appris ;
3. les données de validation et de test utilisent uniquement ces paramètres.

Le nombre de composantes sera configurable et la valeur de reproduction
initiale sera 800, conformément à l'implémentation publique de référence.

### Couches Z-LIN

Ajouter des couches autonomes et testables :

- couche zero-bias/ZAE avec seuil configurable ;
- couche linéaire de goulot d'étranglement ;
- interface commune compatible avec `Sequential`, le backend NumPy/CuPy,
  les gradients et les diagnostics existants.

Les couches préciseront si elles sont pré-entraînables et accepteront un
optimiseur ou un taux d'apprentissage adapté à leur phase.

### Pré-entraînement

Ajouter un orchestrateur indépendant de la boucle supervisée :

- pré-entraînement séquentiel des blocs ;
- conservation des paramètres appris lors de l'ajout du bloc suivant ;
- initialisation et gestion des seuils ZAE conformément au protocole ;
- sortie structurée indiquant les pertes et les paramètres de chaque étape.

Le fine-tuning réutilisera ensuite le modèle complet et l'optimiseur SGD avec
momentum.

### Notebook et expériences

Le notebook séparera clairement :

1. `zlin_baseline_no_augmentation` ;
2. l'expérience augmentée, ajoutée seulement après validation de la baseline ;
3. l'ancien MLP pédagogique.

Chaque expérience affichera son architecture, son prétraitement, ses taux de
dropout, ses learning rates, son nombre d'époques et ses résultats validation
/ test.

## Contrats et vérifications

- aucune statistique PCA ne doit être calculée sur validation ou test ;
- aucune augmentation ne doit être appliquée à la baseline ;
- le test ne doit intervenir qu'après le choix du modèle ;
- les dimensions de chaque couche doivent être vérifiées ;
- le mode évaluation doit désactiver le dropout ;
- le momentum doit conserver un état indépendant par paramètre ;
- les tests doivent couvrir le PCA/whitening, les formes et gradients des
  couches Z-LIN, le pré-entraînement, le mode évaluation et le pipeline
  complet sur un petit jeu synthétique.

## Critère de réussite

Le projet doit produire une expérience identifiée comme baseline Z-LIN,
reproductible avec une graine fixe, dont le protocole est aligné sur l'article.
La valeur publiée de 69,62 % sera une référence de comparaison, pas une
garantie : tout écart devra être attribuable à une différence documentée de
prétraitement, d'optimisation, d'implémentation ou de matériel.
