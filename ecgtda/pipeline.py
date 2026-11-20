"""RR series -> point cloud -> topological features, plus sliding-window analysis."""

import numpy as np
import pandas as pd

from .embedding import takens_embedding
from .hrv import hrv_features
from .topology import topological_features


def rr_cloud(rr, delay=1, dimension=3):
    """Embed an RR series in R^dimension. ``delay`` is in beats."""
    return takens_embedding(rr, delay, dimension)


def tda_features(rr, delay=1, min_persistence=0.005):
    """Topological features of the delay embedding of an RR series."""
    return topological_features(rr_cloud(rr, delay), min_persistence)


def sliding_window_features(rr, window=64, step=4, delay=1):
    """Topological and baseline features on overlapping windows of an RR series.

    Returns a DataFrame indexed by the first beat of each window.
    """
    rr = np.asarray(rr, dtype=float)
    starts = list(range(0, len(rr) - window + 1, step))
    rows = []
    for s in starts:
        w = rr[s:s + window]
        rows.append({**tda_features(w, delay), **hrv_features(w)})
    return pd.DataFrame(rows, index=pd.Index(starts, name="start"))
