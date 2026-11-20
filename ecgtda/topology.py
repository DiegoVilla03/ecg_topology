"""Persistent homology of a point cloud (alpha complex, GUDHI) and summary features."""

import gudhi
import numpy as np

TDA_FEATURES = [
    "h0_max", "h0_mean", "h0_total", "h0_outlier_ratio",
    "h1_count", "h1_max", "h1_mean", "h1_total", "h1_entropy",
]


def persistence(cloud):
    """Persistence diagrams of the alpha filtration of a point cloud.

    GUDHI reports alpha filtration values as *squared* radii; they are converted
    to radii here, so birth and death are in the units of the input (seconds for
    RR clouds). Note that a Vietoris-Rips filtration measures scale as a
    diameter, which is twice the alpha radius for the same event.

    Returns
    -------
    dict
        ``{0: array (m, 2), 1: array (k, 2)}`` with (birth, death) pairs. The
        single infinite H0 class is dropped. Duplicate points are removed first.
    """
    pts = np.unique(np.asarray(cloud, dtype=float), axis=0)
    if len(pts) < 5:
        raise ValueError("need at least 5 distinct points")
    st = gudhi.AlphaComplex(points=pts).create_simplex_tree()
    st.compute_persistence()
    diagrams = {}
    for dim in (0, 1):
        iv = np.asarray(st.persistence_intervals_in_dimension(dim), dtype=float).reshape(-1, 2)
        iv = iv[np.isfinite(iv[:, 1])]
        diagrams[dim] = np.sqrt(iv)
    return diagrams


def persistent_entropy(lifetimes):
    """Shannon entropy of the normalised lifetimes, divided by log(n) so it lies in [0, 1]."""
    lifetimes = np.asarray(lifetimes, dtype=float)
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) < 2:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def topological_features(cloud, min_persistence=0.005):
    """Summary statistics of the H0 and H1 diagrams of a point cloud.

    ``min_persistence`` is the lifetime threshold used to count H1 loops, in the
    units of the cloud (5 ms for RR clouds). It is a modelling choice, not a
    theoretical constant.
    """
    dg = persistence(cloud)
    p0 = dg[0][:, 1] - dg[0][:, 0]
    p1 = dg[1][:, 1] - dg[1][:, 0]
    return {
        "h0_max": float(p0.max()),
        "h0_mean": float(p0.mean()),
        "h0_total": float(p0.sum()),
        "h0_outlier_ratio": float(p0.max() / (np.median(p0) + 1e-12)),
        "h1_count": float((p1 > min_persistence).sum()),
        "h1_max": float(p1.max()) if len(p1) else 0.0,
        "h1_mean": float(p1.mean()) if len(p1) else 0.0,
        "h1_total": float(p1.sum()),
        "h1_entropy": persistent_entropy(p1),
    }
