import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import signal_processor as sp

df = pd.read_csv(r'C:\CODE\Python\ecg_topology\data\raw\MIT_BIH\mitbih_train.csv', header=None)
X = df.iloc[:, :187].values
y = df.iloc[:, 187].values


def save(X_corr, y, filename):
    out = pd.DataFrame(X_corr)
    out['label'] = y.astype(int)
    out.to_csv(filename, index=False, header=False)
    

X_median = sp.median(X)
X_highpass = sp.highpass_filter(X) 
X_diff = sp.differentiation(X)

save(X_median,   y, 'C:\CODE\Python\ecg_topology\data\processed\mitbih_median.csv')
save(X_highpass, y, 'C:\CODE\Python\ecg_topology\data\processed\mitbih_highpass.csv')
save(X_diff,     y, 'C:\CODE\Python\ecg_topology\data\processed\mitbih_diff.csv')