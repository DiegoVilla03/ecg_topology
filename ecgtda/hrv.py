"""Classical HRV features, used as the baseline to compare topological features against."""

import numpy as np

HRV_FEATURES = ["mean_rr", "sdnn", "rmssd", "pnn50", "max_rr_ratio"]


def hrv_features(rr):
    """Time-domain HRV measures from an RR series in seconds.

    ``max_rr_ratio`` (longest RR over the median RR) is not a standard HRV
    index. It is included as a trivial pause detector so that any gain from
    topological features can be judged against something simple.
    """
    rr = np.asarray(rr, dtype=float)
    d = np.diff(rr)
    return {
        "mean_rr": float(rr.mean()),
        "sdnn": float(rr.std(ddof=1)),
        "rmssd": float(np.sqrt(np.mean(d ** 2))),
        "pnn50": float(100.0 * np.mean(np.abs(d) > 0.05)),
        "max_rr_ratio": float(rr.max() / np.median(rr)),
    }
