<div align="center">

# 🫀 TDA for ECG Signal Processing

### Topological Data Analysis for Time Series
**State of the Art and Mathematical Development**
*Application to ECG Signal Processing for Heart Failure and AV Block Detection*

---

![Doctoral Thesis](https://img.shields.io/badge/Type-Doctoral%20Thesis-4a90d9?style=for-the-badge)
![Department](https://img.shields.io/badge/Dept.-Mathematics%20%26%20Signal%20Processing-6c63ff?style=for-the-badge)
![Year](https://img.shields.io/badge/Year-2026-e74c3c?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research%20Stage-27ae60?style=for-the-badge)

</div>

---

## 📖 Overview

Classical time series analysis (ARIMA, SARIMA, Fourier transforms) struggles with **nonlinear, non-stationary signals** that exhibit chaotic or quasi-periodic behavior. This thesis bridges that gap by combining **algebraic topology** with **signal processing** through Topological Data Analysis (TDA).

> *"Data has shape, shape has meaning, and meaning drives value."*

The central insight: when a time series is embedded into a higher-dimensional phase space, it reveals **topological structures** — loops, voids, and connected components — that are invisible to traditional statistical methods. This work systematically applies that insight to cardiac signals.

---

## 🎯 Research Questions

1. How can **Takens' embedding** be optimally applied to physiological time series?
2. What **topological features** are most informative for cardiac signal classification?
3. Can **persistent homology** detect subtle changes in cardiac dynamics *before* clinical manifestation of heart failure?
4. How can the dimensionality constraint (**d ≤ 3**) be leveraged for clinical interpretability?

---

## 🧠 Key Innovations

| Feature | Description |
|---|---|
| 🔷 **Geometric Interpretation** | Topological features provide intuitive geometric views of complex dynamics |
| 📏 **Multiscale Analysis** | Persistent homology captures structure across *all scales* without arbitrary thresholds |
| 🛡️ **Noise Robustness** | Stability theorems guarantee robustness against perturbations |
| 🌀 **Nonlinear Dynamics** | Naturally handles chaotic and nonlinear behavior |

---

## 📐 Mathematical Framework

```
Univariate ECG Signal
        │
        ▼
┌───────────────────────┐
│  Takens' Embedding    │  τ = 1/3 avg. heartbeat duration
│  R-R series → R³     │  d = 3 (efficiency + interpretability)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Alpha Complex        │  Preferred over Vietoris-Rips in R³
│  Filtration           │  for geometric fidelity
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Persistent Homology  │  H₀ (components), H₁ (loops), H₂ (voids)
│  Computation          │  → Persistence Diagrams
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Feature Extraction   │  Topological indices, persistence measures,
│  + Classification     │  cardiac-specific signatures → SVM / Random Forest
└───────────────────────┘
```

### Core Theoretical Pillars

- **Takens' Theorem (1981)** — Phase space reconstruction from a single observed time series via time-delay embedding
- **Persistent Homology** — Tracks birth and death of topological features across filtration scales
- **Stability Theorem (Cohen-Steiner et al., 2007)** — Guarantees small perturbations in data produce small changes in persistence diagrams (bottleneck/Wasserstein distance)

---

## 🫀 Application: ECG Analysis

### Detection Targets

**Heart Failure**
- Altered autonomic regulation → changes in HRV patterns
- Topological signature: fewer persistent H₁ features, altered persistence value distribution
- Reduced complexity of cardiac attractor

**AV Blocks**
- Blocked beats → characteristic patterns in embedded attractor
- Abnormally long R-R intervals appear as outliers in the point cloud
- Regular cycle structure disruption → new connected components in sublevel set filtration
- Paroxysmal AV blocks: detected via **sliding window analysis** and sudden changes in persistence diagrams

### Methodology Pipeline

```
Step 1 │ PREPROCESSING     → Baseline wander removal, R-peak detection, artifact handling
Step 2 │ EMBEDDING         → R-R intervals embedded in R³ via Takens' theorem
Step 3 │ FILTRATION        → Alpha complex construction from point cloud
Step 4 │ HOMOLOGY          → Persistent homology for H₀ and H₁
Step 5 │ FEATURES          → Topological indices + cardiac-specific measures
Step 6 │ CLASSIFICATION    → SVM / Random Forest on topological feature vectors
```

---

## 📚 Thesis Structure

```
📄 thesis_tda_ecg/
├── 1. Introduction
│   └── Motivation, objectives, research questions
├── 2. State of the Art
│   ├── 2.1  TDA Foundations
│   ├── 2.2  Persistent Homology Theory
│   ├── 2.3  Time Series Analysis via TDA
│   └── 2.4  Applications to Physiological Signals
├── 3. Mathematical Framework
│   ├── 3.1  Takens' Embedding Theorem
│   ├── 3.2  Simplicial Complexes & Filtrations
│   ├── 3.3  Persistent Homology Computation
│   └── 3.4  Stability and Robustness
├── 4. Application to ECG Analysis
│   ├── 4.1  Cardiac Signal Characteristics
│   ├── 4.2  Proposed Methodology
│   └── 4.3  Detection of Heart Failure and AV Blocks
└── 5. Conclusions and Future Work
```

---

## 🔬 Tools & Software

| Library | Purpose |
|---|---|
| [GUDHI](https://gudhi.inria.fr/) | Persistent homology computation |
| [Ripser](https://github.com/Ripser/ripser) | Efficient Vietoris-Rips persistence |
| [Dionysus](https://www.mrzv.org/software/dionysus2/) | Topological data analysis |
| Scikit-learn | SVM & Random Forest classifiers |
| NumPy / SciPy | Signal processing & numerical computation |

---

## 🚀 Future Work

- [ ] Large-scale validation on public ECG databases (PhysioNet, MIT-BIH)
- [ ] Real-time algorithms for online cardiac monitoring
- [ ] Multivariate persistent homology for multi-lead ECG
- [ ] Integration with deep learning (end-to-end classification)
- [ ] Extension to EEG, EMG, and respiratory signals

---

## 📑 Key References

1. Edelsbrunner, Letscher & Zomorodian (2002) — *Topological persistence and simplification*
2. Takens (1981) — *Detecting strange attractors in turbulence*
3. Graff et al. (2021) — *Persistent homology for heart rate variability assessment*
4. Emrani, Gentimis & Krim (2014) — *Persistent homology of delay embeddings for wheeze detection*
5. Cohen-Steiner, Edelsbrunner & Harer (2007) — *Stability of persistence diagrams*
6. Biagetti et al. (2021) — *Heart disease classification based on persistent homology*

> Full bibliography available in the thesis document.

---

## 📄 Citation

```bibtex
@phdthesis{tda_ecg_2026,
  title     = {Topological Data Analysis for Time Series:
               State of the Art and Mathematical Development.
               Application to ECG Signal Processing for
               Heart Failure and AV Block Detection},
  school    = {Department of Mathematics and Signal Processing},
  year      = {2026},
  month     = {March},
  type      = {Doctoral Thesis}
}
```

---

<div align="center">

*© 2026 — Doctoral Thesis, Department of Mathematics and Signal Processing*

**"The intersection of algebraic topology and signal processing represents a fertile ground for innovation."**

</div>