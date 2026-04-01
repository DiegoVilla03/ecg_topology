"""
TDA-ECG Analysis Pipeline
Topological Data Analysis for ECG Signal Classification

This module implements the complete TDA pipeline for analyzing ECG signals
as described in the thesis: "Topological Data Analysis for Time Series"
"""

import numpy as np
import pandas as pd
import gudhi
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================
# TAKENS' EMBEDDING
# ============================================

def takens_embedding(time_series, delay, dimension):
    """
    Perform Takens' time-delay embedding

    Parameters:
    -----------
    time_series : array-like
        Input time series (e.g., RR intervals)
    delay : int
        Time delay (in samples)
    dimension : int
        Embedding dimension

    Returns:
    --------
    embedded : ndarray
        Embedded point cloud of shape (n_points, dimension)
    """
    n = len(time_series)
    n_embedded = n - (dimension - 1) * delay

    if n_embedded <= 0:
        return np.array([]).reshape(0, dimension)

    embedded = np.zeros((n_embedded, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay:i * delay + n_embedded]

    return embedded


# ============================================
# PERSISTENT HOMOLOGY
# ============================================

def compute_alpha_persistence(points):
    """
    Compute persistence diagrams using Alpha Complex

    Parameters:
    -----------
    points : ndarray
        Point cloud data of shape (n_points, dimension)

    Returns:
    --------
    persistence : list
        List of persistence pairs (dimension, (birth, death))
    """
    alpha_complex = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()
    persistence = simplex_tree.persistence()
    return persistence


def extract_diagram(persistence, dimension):
    """
    Extract persistence diagram for a specific dimension

    Parameters:
    -----------
    persistence : list
        Persistence pairs from Gudhi
    dimension : int
        Homology dimension (0, 1, 2, ...)

    Returns:
    --------
    diagram : ndarray
        Points (birth, death) for the given dimension
    """
    return np.array([p[1] for p in persistence if p[0] == dimension])


def compute_bottleneck_distance(diag1, diag2):
    """
    Compute the Bottleneck distance between two persistence diagrams

    Parameters:
    -----------
    diag1, diag2 : ndarray
        Persistence diagrams (birth, death pairs)

    Returns:
    --------
    distance : float
        Bottleneck distance
    """
    if len(diag1) == 0 or len(diag2) == 0:
        return np.inf
    return gudhi.bottleneck_distance(diag1, diag2)


# ============================================
# OPTIMAL DELAY SEARCH
# ============================================

def find_optimal_delay(time_series, dimension=3, max_delay=20, homology_dim=1):
    """
    Find optimal delta t for Takens' embedding using Bottleneck distance stability

    Parameters:
    -----------
    time_series : array-like
        Input time series (e.g., RR intervals)
    dimension : int
        Embedding dimension (Takens)
    max_delay : int
        Maximum delay to test
    homology_dim : int
        Homology dimension to track for stability (usually 1 for loops)

    Returns:
    --------
    delays : list
        List of delays tested
    distances : list
        Bottleneck distances between consecutive delays
    """
    delays = list(range(1, max_delay + 1))
    distances = []
    prev_diag = None

    for tau in delays:
        embedded = takens_embedding(time_series, tau, dimension)
        if len(embedded) < dimension + 1:
            distances.append(np.inf)
            continue

        persistence = compute_alpha_persistence(embedded)
        current_diag = extract_diagram(persistence, homology_dim)

        if prev_diag is not None:
            dist = compute_bottleneck_distance(prev_diag, current_diag)
            distances.append(dist)
        else:
            distances.append(0.0) # Placeholder for first delay

        prev_diag = current_diag

    return delays, distances
