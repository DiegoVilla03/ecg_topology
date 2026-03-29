# TDA-ECG Analysis Pipeline

Complete implementation of Topological Data Analysis for ECG signal classification,
as described in the thesis "Topological Data Analysis for Time Series".

## Overview

This pipeline implements the complete TDA framework for analyzing ECG signals:
1. **ECG Preprocessing** - R-peak detection and RR interval extraction
2. **Takens' Embedding** - Phase space reconstruction (d=3, delay=1)
3. **Persistent Homology** - Vietoris-Rips filtration computation
4. **Feature Extraction** - Topological and HRV features
5. **Classification** - Random Forest classifier

## Files

- `tda_ecg_pipeline.py` - Complete Python implementation
- `ecg_samples.png` - Synthetic ECG samples by class
- `preprocessing.png` - R-peak detection and RR interval analysis
- `takens_embedding.png` - 3D attractor visualization
- `persistence_diagrams.png` - Persistence diagrams (H0, H1, H2)
- `tda_analysis_results.png` - Comprehensive results visualization

## Classes

The pipeline was tested on 4 classes:
- **Normal** - Normal sinus rhythm
- **PVC** - Premature Ventricular Contractions
- **AV_Block** - Atrioventricular blocks
- **Heart_Failure** - Heart failure patterns

## Key Results

- **Accuracy**: 100% on test set (12 samples)
- **Cross-validation**: 85.3% (5-fold)
- **Top Features**:
  1. H1_std_persistence (9.5%)
  2. H0_mean_persistence (7.2%)
  3. H0_mean_death (5.0%)
  4. pnn50 (4.9%)
  5. max_rr (4.9%)

## Usage

```python
from tda_ecg_pipeline import process_ecg_sample

# Process a single ECG sample
features = process_ecg_sample(signal, fs=360, label='Normal')
```

## Dependencies

- numpy
- scipy
- pandas
- scikit-learn
- matplotlib

## Mathematical Framework

### Takens' Embedding
```
Φ(x) = (h(x), h(f(x)), h(f²(x)), ..., h(f^(d-1)(x)))
```

### Persistent Homology
- **H0**: Connected components
- **H1**: Cycles (loops)
- **H2**: Voids (3D structures)

### Features Extracted
- 24 topological features (H0, H1, H2 statistics)
- 7 HRV features (mean_rr, std_rr, rmssd, pnn50, etc.)
- 3 cross-dimensional ratios

## References

See thesis document for complete bibliography.
