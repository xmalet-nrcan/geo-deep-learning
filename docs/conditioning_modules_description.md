# Modules de conditionnement et régularisation pour la détection de changement SAR

## Vue d'ensemble

Le modèle ChangeFormer (Bandara & Patel, IGARSS 2022) est un réseau siamois à base de Transformer conçu pour la détection de changement en télédétection. Dans sa version originale, il traite les images pré- et post-événement de manière symétrique sans tenir compte des métadonnées d'acquisition.

Pour adapter ce modèle à la détection de feux de forêt à partir de données SAR RCM (RADARSAT Constellation Mission), nous avons développé quatre modules complémentaires qui s'insèrent dans le pipeline d'inférence **sans modifier l'architecture interne** du ChangeFormer. Ces modules exploitent les spécificités du signal SAR et les métadonnées d'acquisition disponibles.

### Pipeline de traitement

```
Entrée [B, C, H, W]
    │
    ├── (1) FiLM Conditioner      — Modulation par métadonnées d'acquisition
    ├── (2) CBAM                   — Attention canal + spatiale
    ├── (3) Channel Dropout        — Régularisation par suppression de bandes
    │
    ├── Encodeur Transformer siamois (ChangeFormer)
    ├── Décodeur multi-échelles avec différences de features
    │
    └── (4) DFA                    — Attention sur les features de différence
```

Chaque module est optionnel (activable par flag booléen) et s'initialise proche de l'identité pour permettre le fine-tuning à partir de checkpoints existants.

---

## 1. FiLM — Feature-wise Linear Modulation

### Description

Le module FiLM (Perez et al., AAAI 2018) encode les métadonnées catégorielles d'acquisition (passe satellitaire, faisceau, saison, delta temporel) sous forme d'embeddings appris, puis génère des paramètres de modulation par canal (scale γ et shift β) appliqués aux features d'entrée :

$$x_{conditionné} = \gamma \cdot x + \beta$$

Chaque champ de métadonnée est encodé par un embedding indépendant. Les embeddings sont concaténés puis passés dans un MLP partagé pour produire les paramètres γ et β.

### Motivation

En imagerie SAR, la géométrie d'acquisition a un impact direct sur le signal rétrodiffusé :
- La **passe orbitale** (ascendante/descendante) modifie l'angle d'illumination et donc les ombres radar
- Le **faisceau** (A/B/C/D) détermine l'angle d'incidence (20°–49°) et affecte la réponse des surfaces
- La **saison** influence la teneur en eau de la végétation et du sol
- Le **delta temporel** entre acquisitions pré/post conditionne l'amplitude des changements observés

Sans FiLM, ces métadonnées devaient être encodées comme bandes spatiales constantes, gaspillant de la capacité dans l'encodeur Transformer.

### Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `film_embed_dim` | 32 | Dimension de l'embedding par champ de métadonnée |
| `film_metadata_fields` | auto | Dict {nom_champ: nb_catégories} |

### Impact attendu

- Réduction de 2–3 canaux d'entrée (métadonnées retirées des bandes image)
- Capacité accrue de l'encodeur Transformer (canaux libérés pour les données)
- Adaptation spécifique à la géométrie d'acquisition sans augmenter la complexité du réseau
- Coût en paramètres : < 0.01% du modèle total

---

## 2. CBAM — Convolutional Block Attention Module

### Description

Le CBAM (Woo et al., ECCV 2018) applique séquentiellement deux mécanismes d'attention :

1. **Attention canal** : un MLP traite les statistiques globales (average pooling + max pooling) pour produire un poids par canal, identifiant *quelles bandes* sont les plus informatives.

2. **Attention spatiale** : une convolution sur les statistiques agrégées par canal produit une carte spatiale d'attention, identifiant *où* se concentrer dans l'image.

Le module est appliqué en mode résiduel : `sortie = entrée + CBAM(entrée)`.

### Motivation

Les données Stokes RCM comprennent 9–10 bandes spectrales (S0, Sp1, Sp2, Sp3, PDN, PVN, PSN, NDSV, RFDI, angle d'incidence local) dont l'informativité varie selon le type de changement observé :
- Pour un feu récent : PDN et RFDI sont fortement discriminants
- Pour la repousse : PVN et NDSV dominent
- L'angle d'incidence local modifie la pertinence relative de chaque paramètre

Le CBAM permet au modèle d'apprendre dynamiquement cette pondération.

### Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `cbam_reduction` | 4 | Ratio de compression du bottleneck dans l'attention canal |
| `spatial_kernel` | 7 | Taille du noyau pour l'attention spatiale |

### Impact attendu

- Focalisation automatique sur les bandes discriminantes pour chaque échantillon
- Suppression du signal non-informatif (ex: bandes saturées par le speckle)
- Amélioration de la précision sans augmentation significative du coût computationnel
- Coût en paramètres : ~200 paramètres (négligeable)

---

## 3. Channel Dropout — Suppression stochastique de bandes

### Description

Pendant l'entraînement uniquement, le module supprime aléatoirement des canaux d'entrée entiers avec une probabilité indépendante par canal. Un mécanisme de garantie assure qu'au moins la **moitié des bandes** (⌈C/2⌉, minimum 3) reste active. Le même masque de suppression est appliqué aux images pré et post pour préserver le signal de changement.

Les canaux survivants sont re-normalisés (division par le ratio de canaux actifs) pour maintenir l'espérance statistique du signal.

### Motivation

Le speckle SAR est un bruit multiplicatif qui peut dominer certaines bandes de manière aléatoire. De plus, les paramètres de Stokes étant calculés à partir des mêmes canaux de polarisation (HH, HV, VH, VV), ils présentent des corrélations fortes. Le Channel Dropout :
- Force le modèle à ne pas sur-dépendre d'une seule bande (ex: ne pas s'appuyer uniquement sur S0)
- Crée un ensemble implicite de sous-modèles entraînés sur des sous-ensembles de bandes
- Améliore la robustesse aux bandes bruitées ou manquantes en opération

### Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `channel_dropout_prob` | 0.1 | Probabilité de suppression par canal |
| `min_channels` | ⌈C/2⌉ | Nombre minimum de canaux garantis actifs |

### Impact attendu

- Régularisation spécifique au domaine SAR (analogue au Dropout classique mais sur les bandes d'entrée)
- Réduction du sur-apprentissage sur les bandes dominantes
- Robustesse accrue en conditions opérationnelles (bandes dégradées, artefacts)
- Synergie avec CBAM : empêche l'attention canal de se sur-spécialiser
- Coût : aucun paramètre additionnel (masque stochastique)

---

## 4. DFA — Difference Feature Attention

### Description

Le décodeur ChangeFormer produit 5 prédictions à différentes échelles (4 intermédiaires + 1 finale). Le module DFA applique un **gating appris** sur chaque prédiction intermédiaire : un petit MLP (AdaptiveAvgPool → Linear → ReLU → Linear → Sigmoid) calcule un poids scalaire par tête de décodeur, permettant de supprimer les prédictions peu fiables.

La tête finale n'est jamais gatée (poids fixé à 1).

### Motivation

Dans le contexte SAR, les prédictions à fine échelle (haute résolution spatiale) sont particulièrement sensibles au speckle, générant des faux positifs ponctuels. À l'inverse, les prédictions à échelle grossière capturent mieux les structures spatiales des feux mais perdent la précision des contours. Le DFA permet au modèle d'apprendre automatiquement la fiabilité relative de chaque échelle selon le contenu de la scène.

### Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `dfa_gate_hidden` | 16 | Dimension cachée du MLP de gating par échelle |
| `num_scales` | 4 | Nombre de têtes intermédiaires du décodeur |

### Impact attendu

- Réduction des faux positifs issus du speckle aux échelles fines
- Amélioration de la deep supervision en pondérant dynamiquement les pertes auxiliaires
- Adaptation à la taille des feux : les petits feux bénéficient des échelles fines, les grands feux des échelles grossières
- Coût en paramètres : ~300 paramètres (4 petits MLP)

---

## Tableau récapitulatif

| Module | Point d'insertion | Paramètres ajoutés | Inférence | Cible | Bénéfice principal |
|--------|-------------------|-------------------|-----------|-------|-------------------|
| **FiLM** | Avant encodeur | ~3 000 (<0.01%) | ✓ | Métadonnées d'acquisition | Adaptation géométrique sans bandes supplémentaires |
| **CBAM** | Avant encodeur | ~200 (<0.001%) | ✓ | Bandes + espace | Focalisation sur les bandes/régions discriminantes |
| **Channel Dropout** | Avant encodeur | 0 | ✗ (train only) | Bandes d'entrée | Régularisation SAR, robustesse aux bandes bruitées |
| **DFA** | Après décodeur | ~300 (<0.001%) | ✓ | Sorties multi-échelles | Suppression des faux positifs de speckle |

## Tableau des interactions entre modules

| Interaction | Effet |
|-------------|-------|
| FiLM + CBAM | FiLM adapte le signal à la géométrie → CBAM affine la sélection de bandes. Complémentaires. |
| CBAM + Channel Dropout | Le dropout empêche CBAM de concentrer l'attention sur trop peu de bandes → attention plus distribuée et robuste. |
| FiLM + Channel Dropout | FiLM modifie les échelles/offsets puis certaines bandes modulées sont droppées → le FiLM apprend des modulations utiles pour *toutes* les bandes. |
| Deep Supervision + DFA | La deep supervision entraîne toutes les têtes, le DFA apprend lesquelles sont fiables → optimisation conjointe du contenu et de la confiance des prédictions intermédiaires. |

## Références

- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
- Woo, S., Park, J., Lee, J.Y., & Kweon, I.S. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
- Bandara, W.G.C., & Patel, V.M. (2022). A Transformer-Based Siamese Network for Change Detection. *IGARSS*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.
