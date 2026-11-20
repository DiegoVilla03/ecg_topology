"""Time-delay (Takens) embedding and delay selection."""

import numpy as np


def takens_embedding(time_series, delay, dimension):
    """Time-delay embedding of a scalar series.

    Row i of the result is (x[i], x[i + delay], ..., x[i + (dimension - 1) * delay]).

    Parameters
    ----------
    time_series : array-like, shape (n,)
        Scalar series. For the RR pipeline this is the RR-interval series, so
        ``delay`` is measured in beats, not in seconds or samples.
    delay : int
        Lag between coordinates (in samples of ``time_series``).
    dimension : int
        Embedding dimension.

    Returns
    -------
    ndarray, shape (n - (dimension - 1) * delay, dimension)
        Empty array if the series is too short.
    """
    x = np.asarray(time_series, dtype=float)
    n_embedded = len(x) - (dimension - 1) * delay
    if n_embedded <= 0:
        return np.empty((0, dimension))
    return np.column_stack([x[i * delay:i * delay + n_embedded] for i in range(dimension)])


def mutual_information(x, lag, bins=8):
    """Histogram estimate of the mutual information between x[t] and x[t + lag]."""
    x = np.asarray(x, dtype=float)
    joint, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=bins)
    pxy = joint / joint.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


def choose_delay(x, max_delay=10, bins=8):
    """First local minimum of the mutual information, a standard choice for the delay.

    Falls back to the lag with the smallest mutual information when the curve
    has no interior minimum. Histogram estimates are noisy for short series
    (a few hundred points), so treat the result as a suggestion.
    """
    mi = [mutual_information(x, lag, bins) for lag in range(1, max_delay + 1)]
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i - 1] and mi[i] <= mi[i + 1]:
            return i + 1
    return int(np.argmin(mi)) + 1
