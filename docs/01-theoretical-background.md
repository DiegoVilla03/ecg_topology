# 1. Theoretical background

Source material: `thesis_bg/thesis_tda_ecg.docx`, chapters 2 and 3, condensed and checked against the cited literature. References are in [references.md](references.md).

## 1.1 Why topology for time series

Linear tools (ARIMA-type models, spectral analysis) describe a series through correlations and frequencies. They say little about the *shape* of the dynamics. Topological data analysis (TDA) measures shape directly: connected components, loops and voids of a point cloud, tracked across all spatial scales. A time series becomes a point cloud through a delay embedding (section 1.3), and a quasi-periodic signal then leaves a loop in the cloud.

## 1.2 Persistent homology

Take a nested family of spaces, a **filtration**

$$\emptyset = X_0 \subseteq X_1 \subseteq \dots \subseteq X_n = X .$$

Homology $H_k$ counts $k$-dimensional holes: $H_0$ connected components, $H_1$ loops, $H_2$ voids. As the filtration grows, a hole is **born** at scale $b$ and **dies** at scale $d$. The multiset of pairs $(b, d)$ is the **persistence diagram**, and $d - b$ is the **lifetime**. Long lifetimes are read as structure, short ones as noise. This reading is a heuristic; the stability theorem (1.5) is what justifies it quantitatively.

Computation reduces to a matrix reduction over the simplices of the filtration. The standard algorithm is cubic in the number of simplices in the worst case, and is much faster in practice. GUDHI, Ripser and Dionysus provide implementations; this repository uses GUDHI.

## 1.3 Delay embedding (Takens)

For a smooth map $f: M \to M$ on a compact manifold of dimension $m$ and a generic smooth observable $h: M \to \mathbb{R}$, the delay map

$$\Phi(x) = \big(h(x),\, h(f(x)),\, \dots,\, h(f^{\,d-1}(x))\big)$$

is an embedding when $d \ge 2m + 1$ (Takens, 1981). In practice a scalar series $x_1, \dots, x_n$ is turned into the cloud

$$y_i = \big(x_i,\, x_{i+\tau},\, \dots,\, x_{i+(d-1)\tau}\big), \qquad i = 1, \dots, n - (d-1)\tau .$$

Two parameters must be chosen:

- **Delay $\tau$.** Usually the first minimum of the average mutual information, or the first zero of the autocorrelation. `ecgtda.embedding.choose_delay` implements the first.
- **Dimension $d$.** Usually the false-nearest-neighbours criterion. Fixing $d = 3$, as this project does, is a pragmatic choice for visualisation and cost. It does not satisfy Takens' bound for a general attractor, so it has to be justified empirically (sensitivity of the results to $d$).

## 1.4 Filtrations on a point cloud

**Vietoris–Rips.** $\mathrm{VR}(X, \varepsilon)$ contains every finite subset of $X$ with diameter at most $\varepsilon$. It works in any metric space and is expensive in high dimension.

**Alpha complex.** The subcomplex of the Delaunay triangulation made of simplices whose smallest enclosing ball has radius at most $\alpha$. It is homotopy equivalent to the union of balls of radius $\alpha$ around the points, so it carries the same homotopy type as the Čech complex, and it is small in $\mathbb{R}^2$ and $\mathbb{R}^3$. That is why it is the choice for $d \le 3$.

Two conventions to keep in mind when comparing numbers:

- The alpha complex measures scale as a **radius**, Vietoris–Rips as a **diameter**. The same event appears at half the value in the alpha filtration.
- GUDHI stores alpha filtration values as **squared** radii. `ecgtda.topology.persistence` takes the square root, so diagrams are in the units of the data (seconds for RR clouds).

Alpha and Vietoris–Rips filtrations are not interchangeable in general. They agree closely on well-sampled clouds, which is a reason to cross-check, not a theorem.

## 1.5 Stability

Let $f, g$ be tame functions. The bottleneck distance between the persistence diagrams of their sublevel-set filtrations satisfies

$$d_B\big(\mathrm{Dgm}(f), \mathrm{Dgm}(g)\big) \le \lVert f - g \rVert_\infty$$

(Cohen-Steiner, Edelsbrunner and Harer, 2007). For point clouds, filtrations built from distances are stable with respect to the Gromov–Hausdorff distance, with a constant that depends on the construction; for Vietoris–Rips and Čech filtrations it is 2 (Chazal, de Silva and Oudot, 2014):

$$d_B\big(\mathrm{Dgm}(X), \mathrm{Dgm}(Y)\big) \le 2\, d_{GH}(X, Y).$$

Consequences for ECG work:

- Small measurement noise moves points of the diagram a small distance. Features that change *abruptly* are therefore informative.
- Stability does not make every feature robust. Quantities such as "number of points with lifetime above a threshold" are not continuous in the diagram, so they can jump under small perturbations. Prefer distances between diagrams, total persistence or persistent entropy when robustness matters.
- Bottleneck distance is driven by the single largest discrepancy. Wasserstein distance is more sensitive to the whole diagram.

## 1.6 Summary statistics used in this repository

| Feature | Definition |
|---|---|
| `h0_max`, `h0_mean`, `h0_total` | max, mean, sum of finite $H_0$ lifetimes |
| `h0_outlier_ratio` | max $H_0$ lifetime over the median lifetime |
| `h1_count` | number of $H_1$ classes with lifetime above `min_persistence` (default 5 ms) |
| `h1_max`, `h1_mean`, `h1_total` | max, mean, sum of $H_1$ lifetimes |
| `h1_entropy` | persistent entropy of $H_1$ lifetimes, $-\sum p_i \log p_i / \log n$ with $p_i = \ell_i / \sum_j \ell_j$ |

`min_persistence` is a modelling choice with the units of the cloud. Report results over a range of values, not one.
