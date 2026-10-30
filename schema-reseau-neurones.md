## Neurone unique

```mermaid
flowchart TD
  I["Image"] --> V["Vectorisation"]
  V --> X["x"]
  W["w"] --> Z["Score z"]
  B["b"] --> Z
  X --> Z
  Z --> P["Probabilite p"]
  Y["Cible y"] --> J["Cout J(w,b)"]
  P --> J
  J --> G["Gradients"]
  G --> U1["Update de w"]
  G --> U2["Update de b"]
  U1 -.-> W
  U2 -.-> B
```

$$
z = w^\top x + b
$$

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
p = \mathbb{P}(y=1 \mid x)
$$

$$
\mathbb{P}(y \mid x; w,b) = p^y (1-p)^{1-y}
$$

$$
\mathcal{L}(w,b) = \prod_{i=1}^{m} \mathbb{P}(y_i \mid x_i; w,b)
$$

$$
\log \mathcal{L}(w,b) = \sum_{i=1}^{m} \log \mathbb{P}(y_i \mid x_i; w,b)
$$

$$
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
$$

$$
\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(p_i - y_i)x_i
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(p_i - y_i)
$$

$$
w \leftarrow w - \eta \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \eta \frac{\partial J}{\partial b}
$$

## Forme vectorielle

$$
X \in \mathbb{R}^{m \times d},
\qquad
w \in \mathbb{R}^{d},
\qquad
b \in \mathbb{R}
$$

$$
z = Xw + b\mathbf{1}
$$

$$
p = \sigma(z)
$$

$$
J(w,b) = -\frac{1}{m}\left[y^\top \log p + (1-y)^\top \log(1-p)\right]
$$

$$
\nabla_w J = \frac{1}{m}X^\top(p-y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\mathbf{1}^\top(p-y)
$$

## Reseau dense

```mermaid
flowchart TD
  X0["Entree x"] --> D1["Dense 1"]
  D1 --> A1["Activation 1"]
  A1 --> D2["Dense 2"]
  D2 --> A2["Activation 2"]
  A2 --> AL["Sortie a[L]"]
  YL["Cible y"] --> JL["Cout J"]
  AL --> JL
  JL --> BP["Backprop"]
  BP --> UP["Mise a jour des couches"]
```

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$

$$
a^{[l]} = g^{[l]}(z^{[l]})
$$

$$
dz^{[L]} = a^{[L]} - y
$$

$$
dW^{[l]} = \frac{1}{m}\,dz^{[l]}(a^{[l-1]})^\top
$$

$$
db^{[l]} = \frac{1}{m}\sum_{i=1}^{m}dz_i^{[l]}
$$

$$
da^{[l-1]} = (W^{[l]})^\top dz^{[l]}
$$

## Forme vectorielle dense

Pour un batch de taille $m$ :

$$
A^{[0]} = X \in \mathbb{R}^{d_0 \times m}
$$

$$
W^{[l]} \in \mathbb{R}^{d_l \times d_{l-1}},
\qquad
b^{[l]} \in \mathbb{R}^{d_l \times 1}
$$

$$
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}\mathbf{1}^\top
$$

$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

$$
dZ^{[L]} = A^{[L]} - Y
$$

$$
dW^{[l]} = \frac{1}{m} dZ^{[l]} {A^{[l-1]}}^\top
$$

$$
db^{[l]} = \frac{1}{m} \sum_{j=1}^{m} dZ^{[l]}_{:,j}
$$

$$
dA^{[l-1]} = {W^{[l]}}^\top dZ^{[l]}
$$

Logistic regression correspond au cas $L = 1$.

## Mini-batchs

On découpe les $m$ exemples en lots de taille $B$ :

$$
\mathcal{B}_k \subset \{1,\dots,m\},
\qquad
|\mathcal{B}_k| = B
$$

Le gradient est calculé sur chaque lot :

$$
dW_{\mathcal{B}_k}^{[l]}
= \frac{1}{B}\,dZ_{\mathcal{B}_k}^{[l]}
{A_{\mathcal{B}_k}^{[l-1]}}^\top
$$

$$
db_{\mathcal{B}_k}^{[l]}
= \frac{1}{B}\sum_{i\in\mathcal{B}_k} dz_i^{[l]}
$$

Puis les paramètres sont mis à jour :

$$
W^{[l]} \leftarrow W^{[l]} - \eta\,dW_{\mathcal{B}_k}^{[l]}
$$

$$
b^{[l]} \leftarrow b^{[l]} - \eta\,db_{\mathcal{B}_k}^{[l]}
$$

## Régularisation L2

La fonction de coût régularisée pénalise les poids trop grands :

$$
J_{\mathrm{total}}
= J_{\mathrm{classification}}
+ \frac{\lambda}{2}
\sum_{l=1}^{L}\left\|W^{[l]}\right\|_F^2
$$

La pénalité ne s'applique pas aux biais :

$$
\frac{\partial J_{\mathrm{total}}}{\partial W^{[l]}}
= dW^{[l]} + \lambda W^{[l]}
$$

$$
\frac{\partial J_{\mathrm{total}}}{\partial b^{[l]}}
= db^{[l]}
$$

## Fonctions d'activation

```mermaid
flowchart LR
  ZH["Sortie couche cachée z"] --> R["ReLU"]
  R --> AH["a = max(0, z)"]
  ZL["Scores de sortie z"] --> S["Softmax"]
  S --> P["Probabilités p"]
  ZB["Score binaire z"] --> G["Sigmoïde"]
  G --> PB["Probabilité p"]
```

### ReLU

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

$$
\operatorname{ReLU}'(z)=
\begin{cases}
0 & z \le 0\\
1 & z > 0
\end{cases}
$$

### Sigmoïde

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

$$
\sigma'(z)=\sigma(z)\left(1-\sigma(z)\right)
$$

### Softmax

Pour $K$ classes :

$$
p_k=\operatorname{softmax}(z)_k
=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

$$
\sum_{k=1}^{K}p_k=1
$$

## Early stopping

```mermaid
flowchart TD
  E["Fin d'une époque"] --> V["Calculer la loss validation"]
  V --> B{"Amélioration ?"}
  B -- Oui --> S["Sauvegarder le modèle"]
  B -- Non --> P["Incrémenter patience"]
  P --> L{"Patience atteinte ?"}
  L -- Non --> E
  L -- Oui --> R["Arrêter et restaurer le meilleur modèle"]
```

Avec une patience $p$ :

$$
\text{arrêt si aucune amélioration pendant }p\text{ époques}
$$
