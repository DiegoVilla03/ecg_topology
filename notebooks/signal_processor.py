import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def median(X):
    X_median = X - np.median(X, axis=1, keepdims=True)
    return X_median
    
def highpass_filter(X, cutoff=0.5, fs=125, order=4):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, X, axis=1)

def differentiation(X):
    X_diff = np.diff(X, axis=1)
    X_diff = np.concatenate([np.zeros((X.shape[0], 1)), np.diff(X, axis=1)], axis=1)
    return X_diff