# 2. Methodology

Source material: `thesis_bg/thesis_tda_ecg.docx`, chapter 4. This document states what the pipeline does, which decisions are still open, and how it will be validated.

## 2.1 Detection targets

**Heart failure.** Reduced and less structured heart rate variability. Hypothesis: the embedded RR cloud loses its loop structure, so fewer or shorter-lived $H_1$ classes.

**Atrioventricular (AV) block.** Conduction from atria to ventricles is delayed or interrupted.

| Type | What happens | Visible in the RR series? |
|---|---|---|
| First degree | PR interval prolonged, every beat conducted | **No.** RR is unchanged |
| Second degree, type I (Wenckebach) | PR lengthens until a beat is dropped | Yes: RR shortens over the cycle, then a pause |
| Second degree, type II | Intermittent dropped beats, constant PR | Yes: isolated pauses of about two atrial intervals |
| Paroxysmal | Sudden run of non-conducted beats | Yes: a pause of several seconds |
| Third degree | Atria and ventricles dissociated | The ventricular escape rhythm is typically slow and regular |

The first row matters: **an RR-based pipeline cannot detect first-degree block**, and a regular escape rhythm may look benign to it. Whether the project targets all AV blocks or only those that change the RR series has to be stated explicitly in the proposal.

## 2.2 Pipeline

| Step | Operation | Code |
|---|---|---|
| 1 | Preprocessing: baseline removal, R-peak detection, artefact and ectopic-beat handling, RR extraction | not implemented (synthetic RR is generated directly) |
| 2 | Delay embedding of the RR series in $\mathbb{R}^3$ | `ecgtda.embedding.takens_embedding` |
| 3 | Alpha filtration of the cloud | `ecgtda.topology.persistence` |
| 4 | Persistent homology, $H_0$ and $H_1$ | `ecgtda.topology.persistence` |
| 5 | Features from the diagrams | `ecgtda.topology.topological_features` |
| 6 | Classification (SVM, random forest) or sliding-window change detection | `scripts/run_demo.py`, `ecgtda.pipeline.sliding_window_features` |

Heart failure is classified from one window per subject. Paroxysmal block is handled with overlapping windows (typically 64 beats, step 4) and a score per window.

## 2.3 Open design decisions

1. **What is embedded.** The thesis draft sets the delay to one third of the mean heartbeat duration and also embeds "the R-R series". These are two different objects:
   - Embedding the **ECG waveform**: the delay is a time (or number of samples), and one third of a beat is a sensible rule because the cloud then traces the P-QRS-T loop.
   - Embedding the **RR series**: the delay is a number of *beats*; the unit is not time. The current code and demo use this, with delay 1.

   These capture different physiology (waveform morphology versus rhythm) and need different preprocessing. The proposal should pick one as the main line and treat the other as an extension.
2. **Dimension.** $d = 3$ is a choice, not a consequence of Takens (see 1.3). Report sensitivity to $d$.
3. **Scale normalisation.** Features are computed in raw seconds, so they carry amplitude information (a low-variability heart gives short lifetimes). Normalising the cloud would remove that. Which is wanted depends on the hypothesis being tested.
4. **$H_1$ threshold.** `min_persistence` is arbitrary; sweep it.
5. **What topology adds.** Pauses from AV block are also caught by a threshold on the longest RR (see the demo, figure 5). The claim that topology is *complementary* to HRV must be tested on real data against HRV features and simple detectors, not assumed.

## 2.4 Validation plan

Real data, in this order:

1. **Congestive Heart Failure RR Interval Database** (PhysioNet `chf2db`, 29 long-term recordings of subjects with NYHA class I to III, beat annotations). Paired with a normal-rhythm RR database for the control class. The BIDMC Congestive Heart Failure Database holds the digitised ECG.
2. **MIT-BIH Arrhythmia Database** for annotated rhythm segments, including conduction abnormalities. Its record list needs to be checked for enough AV block examples before committing to it.
3. A source with annotated **paroxysmal AV block** is still to be identified; this is the hardest target and may need to be dropped or reframed.

Rules for the evaluation:

- Split **by subject**, never by window. Windows from one recording are strongly correlated.
- Report sensitivity, specificity and ROC curves with confidence intervals, not accuracy alone.
- Always include two baselines: classical HRV (SDNN, RMSSD, pNN50, and frequency-domain measures) and, for pauses, a longest-RR threshold.
- Handle ectopic beats explicitly (exclude, interpolate, or model them) and report how.
- Fix all thresholds and hyperparameters on training subjects only.
