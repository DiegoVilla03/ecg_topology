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
# ECG PREPROCESSING
# ============================================

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """Apply bandpass filter to ECG signal"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks(signal, fs, distance_ms=300):
    """
    Detect R-peaks in ECG signal using Pan-Tompkins inspired algorithm

    Parameters:
    -----------
    signal : array-like
        ECG signal
    fs : float
        Sampling frequency in Hz
    distance_ms : float
        Minimum distance between R-peaks in milliseconds

    Returns:
    --------
    r_peaks : ndarray
        Indices of detected R-peaks
    """
    filtered = bandpass_filter(signal, fs)

    # Differentiate
    diff = np.diff(filtered)
    diff = np.append(diff, diff[-1])

    # Square
    squared = diff ** 2

    # Moving window integration
    window_size = int(0.08 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    # Find peaks
    min_distance = int(distance_ms * fs / 1000)
    peaks, _ = find_peaks(integrated, distance=min_distance, 
                          height=np.percentile(integrated, 30))

    # Refine peaks on original signal
    refined_peaks = []
    search_window = int(0.05 * fs)
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)
        local_max = start + np.argmax(signal[start:end])
        refined_peaks.append(local_max)

    return np.array(refined_peaks)


def extract_rr_intervals(r_peaks, fs):
    """Extract RR intervals from R-peak locations in milliseconds"""
    if len(r_peaks) < 2:
        return np.array([])
    rr_samples = np.diff(r_peaks)
    rr_intervals = rr_samples / fs * 1000
    return rr_intervals


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

class VietorisRipsPersistence:
    """
    Vietoris-Rips persistent homology computation

    Implements a simplified but functional version of persistent
    homology for point cloud data.
    """

    def __init__(self, max_dimension=2):
        self.max_dimension = max_dimension

    def compute_persistence(self, points, max_edge_length=None, n_bins=20):
        """
        Compute persistent homology for Vietoris-Rips filtration

        Parameters:
        -----------
        points : ndarray
            Point cloud of shape (n_points, dimension)
        max_edge_length : float, optional
            Maximum edge length for filtration
        n_bins : int
            Number of threshold bins

        Returns:
        --------
        persistence : dict
            Dictionary with homology dimensions as keys and 
            lists of (birth, death) tuples as values
        """
        if len(points) < 3:
            return {0: [], 1: [], 2: []}

        dist_matrix = squareform(pdist(points))
        n_points = len(points)

        if max_edge_length is None:
            max_edge_length = np.percentile(dist_matrix[dist_matrix > 0], 95)

        thresholds = np.linspace(0, max_edge_length, n_bins)

        # Track Betti numbers
        betti_history = {0: [], 1: [], 2: []}

        for eps in thresholds:
            adj = (dist_matrix <= eps) & (dist_matrix > 0)
            np.fill_diagonal(adj, False)
            betti = self._compute_betti_numbers(adj, dist_matrix, eps)
            for dim in [0, 1, 2]:
                betti_history[dim].append((eps, betti[dim]))

        # Extract persistence intervals
        persistence = {0: [], 1: [], 2: []}

        for dim in [0, 1, 2]:
            history = betti_history[dim]
            active_features = {}
            feature_id = 0

            for i, (eps, betti) in enumerate(history):
                if i == 0:
                    for _ in range(betti):
                        active_features[feature_id] = 0
                        feature_id += 1
                else:
                    prev_betti = history[i-1][1]

                    if betti > prev_betti:
                        for _ in range(betti - prev_betti):
                            active_features[feature_id] = eps
                            feature_id += 1
                    elif betti < prev_betti:
                        for _ in range(prev_betti - betti):
                            if active_features:
                                oldest_id = min(active_features.keys())
                                birth = active_features.pop(oldest_id)
                                if eps > birth:
                                    persistence[dim].append((birth, eps))

            for birth in active_features.values():
                persistence[dim].append((birth, max_edge_length))

        return persistence

    def _compute_betti_numbers(self, adj, dist_matrix, eps):
        """Compute Betti numbers for a given adjacency matrix"""
        n_points = len(adj)

        # β₀: Connected components
        n_components, _ = connected_components(csr_matrix(adj), directed=False)
        beta_0 = n_components

        # Count edges
        n_edges = np.sum(adj) // 2

        # β₁: First Betti number
        beta_1 = max(0, n_edges - n_points + n_components)

        # β₂: Count tetrahedra for 3D
        beta_2 = 0
        if len(dist_matrix) >= 4:
            tetra = self._count_tetrahedra(dist_matrix, eps)
            beta_2 = max(0, tetra)

        return {0: beta_0, 1: beta_1, 2: beta_2}

    def _count_tetrahedra(self, dist_matrix, eps):
        """Count 4-cliques (tetrahedra)"""
        n = len(dist_matrix)
        adj = dist_matrix <= eps
        np.fill_diagonal(adj, True)

        count = 0
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a, b]:
                    continue
                for c in range(b+1, n):
                    if not (adj[a, c] and adj[b, c]):
                        continue
                    for d in range(c+1, n):
                        if adj[a, d] and adj[b, d] and adj[c, d]:
                            count += 1
        return count


# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_topological_features(persistence, max_edge_length):
    """
    Extract statistical features from persistence diagrams

    Parameters:
    -----------
    persistence : dict
        Dictionary with homology dimensions as keys
    max_edge_length : float
        Maximum edge length used in filtration

    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    features = {}

    for dim in [0, 1, 2]:
        intervals = persistence[dim]
        prefix = f'H{dim}_'

        if len(intervals) == 0:
            for suffix in ['count', 'mean_persistence', 'max_persistence', 
                          'std_persistence', 'entropy', 'snr', 'mean_birth', 'mean_death']:
                features[f'{prefix}{suffix}'] = 0
            continue

        persistences = [d - b for b, d in intervals if d > b]

        if len(persistences) == 0:
            for suffix in ['count', 'mean_persistence', 'max_persistence', 
                          'std_persistence', 'entropy', 'snr', 'mean_birth', 'mean_death']:
                features[f'{prefix}{suffix}'] = 0
            continue

        features[f'{prefix}count'] = len(persistences)
        features[f'{prefix}mean_persistence'] = np.mean(persistences)
        features[f'{prefix}max_persistence'] = np.max(persistences)
        features[f'{prefix}std_persistence'] = np.std(persistences)

        # Persistence entropy
        if np.sum(persistences) > 0:
            probs = np.array(persistences) / np.sum(persistences)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            features[f'{prefix}entropy'] = entropy
        else:
            features[f'{prefix}entropy'] = 0

        # Signal-to-noise ratio
        threshold = 0.1 * max_edge_length
        persistent = sum(1 for p in persistences if p > threshold)
        non_persistent = len(persistences) - persistent
        features[f'{prefix}snr'] = persistent / (non_persistent + 1)

        births = [b for b, d in intervals]
        deaths = [d for b, d in intervals]
        features[f'{prefix}mean_birth'] = np.mean(births)
        features[f'{prefix}mean_death'] = np.mean(deaths)

    # Cross-dimensional features
    features['H1_H0_ratio'] = features['H1_count'] / (features['H0_count'] + 1)
    features['H2_H1_ratio'] = features['H2_count'] / (features['H1_count'] + 1)
    features['total_features'] = features['H0_count'] + features['H1_count'] + features['H2_count']

    return features


def extract_hrv_features(rr_intervals):
    """Extract traditional HRV features"""
    features = {}
    features['mean_rr'] = np.mean(rr_intervals)
    features['std_rr'] = np.std(rr_intervals)
    features['min_rr'] = np.min(rr_intervals)
    features['max_rr'] = np.max(rr_intervals)
    features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    features['nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    features['pnn50'] = features['nn50'] / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
    return features


# ============================================
# COMPLETE PIPELINE
# ============================================

def process_ecg_sample(signal, fs, label=None):
    """
    Complete TDA pipeline for a single ECG sample

    Parameters:
    -----------
    signal : array-like
        ECG signal
    fs : float
        Sampling frequency
    label : str, optional
        Class label

    Returns:
    --------
    features : dict or None
        Extracted features or None if processing fails
    """
    # Step 1: Detect R-peaks
    r_peaks = detect_r_peaks(signal, fs)
    if len(r_peaks) < 10:
        return None

    # Step 2: Extract RR intervals
    rr_intervals = extract_rr_intervals(r_peaks, fs)
    if len(rr_intervals) < 5:
        return None

    # Step 3: Takens' embedding (d=3, delay=1)
    embedded = takens_embedding(rr_intervals, delay=1, dimension=3)
    if len(embedded) < 5:
        return None

    # Step 4: Persistent homology
    vr = VietorisRipsPersistence()
    max_edge_length = np.percentile(
        squareform(pdist(embedded))[squareform(pdist(embedded)) > 0], 95
    )
    persistence = vr.compute_persistence(embedded, max_edge_length=max_edge_length, n_bins=20)

    # Step 5: Feature extraction
    features = extract_topological_features(persistence, max_edge_length)
    features.update(extract_hrv_features(rr_intervals))

    if label:
        features['label'] = label

    return features


if __name__ == '__main__':
    print("TDA-ECG Analysis Pipeline")
    print("Import this module to use the analysis functions")
