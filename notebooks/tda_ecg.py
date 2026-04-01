"""
TDA-ECG Analysis Pipeline
Topological Data Analysis for ECG Signal Classification

This module implements the complete TDA pipeline for analyzing ECG signals
as described in the thesis: "Topological Data Analysis for Time Series"
"""

import numpy as np
import pandas as pd
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
