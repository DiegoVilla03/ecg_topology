"""Synthetic RR-interval generators for demonstrating and testing the pipeline.

The model has two stages.

1. Atrial rhythm: P-wave times with a mean interval plus a respiratory
   oscillation (~0.25 Hz), a slower Mayer-wave oscillation (~0.1 Hz) and
   Gaussian noise. The ``chf`` profile has a faster mean rate and much smaller
   oscillations.
2. AV conduction: each P wave either reaches the ventricles after a PR delay
   and produces an R peak, or is blocked. The conduction pattern defines the
   rhythm class.

Kinds
-----
nsr      normal sinus rhythm, 1:1 conduction.
chf      reduced-variability profile, 1:1 conduction (caricature of heart failure).
av1      first-degree AV block: constant prolonged PR, 1:1 conduction. The RR
         series is statistically identical to ``nsr``; used as a negative control.
mobitz1  second-degree type I (Wenckebach): PR grows by shrinking increments
         until a P wave is blocked, then resets.
mobitz2  second-degree type II: constant PR, isolated blocked P waves.
pavb     paroxysmal AV block: normal conduction with episodes of several
         consecutive blocked P waves (pauses of a few seconds).

This is a toy model. It encodes the features the thesis hypothesises (less
variability in heart failure, pauses in AV block), so results obtained on it
test the code, not the clinical hypotheses.
"""

from dataclasses import dataclass

import numpy as np

KINDS = ("nsr", "chf", "av1", "mobitz1", "mobitz2", "pavb")

# Per-subject parameter ranges (seconds, except frequencies in Hz).
_PROFILES = {
    "nsr": dict(mean_pp=(0.75, 1.00), rsa=(0.020, 0.060), lf=(0.008, 0.035), noise=(0.004, 0.012)),
    "chf": dict(mean_pp=(0.60, 0.85), rsa=(0.003, 0.025), lf=(0.002, 0.015), noise=(0.004, 0.015)),
}
_PR_JITTER = 0.003


@dataclass
class RRSeries:
    kind: str
    rr: np.ndarray           # ventricular RR intervals, seconds
    r_times: np.ndarray      # R-peak times, seconds (len(rr) + 1)
    pause_mask: np.ndarray   # True where the RR interval spans at least one blocked P wave
    params: dict


def _atrial_times(n, rng, profile):
    p = {k: rng.uniform(*v) for k, v in _PROFILES[profile].items()}
    rsa_hz, lf_hz = rng.uniform(0.20, 0.30), rng.uniform(0.08, 0.12)
    phi_r, phi_l = rng.uniform(0, 2 * np.pi, size=2)
    t, out = 0.0, np.empty(n)
    for k in range(n):
        out[k] = t
        pp = (p["mean_pp"]
              + p["rsa"] * np.sin(2 * np.pi * rsa_hz * t + phi_r)
              + p["lf"] * np.sin(2 * np.pi * lf_hz * t + phi_l)
              + rng.normal(0, p["noise"]))
        t += max(pp, 0.3)
    return out, p


def _conduct(p_times, kind, rng, blocked_p=None):
    """Return (R times, index of the P wave behind each R, params)."""
    n = len(p_times)
    pr_base = rng.uniform(0.24, 0.34) if kind == "av1" else rng.uniform(0.14, 0.20)
    params = {"pr_base": pr_base}
    r_times, r_from = [], []

    if kind == "mobitz1":
        gain = rng.uniform(0.10, 0.22)       # total PR prolongation within a cycle
        params["gain"] = gain
        i, cycle = 0, int(rng.integers(3, 6))
        for k in range(n):
            if i == cycle - 1:               # blocked P wave: reset the cycle
                i, cycle = 0, int(rng.integers(3, 6))
                continue
            pr = pr_base + gain * (1 - np.exp(-i / 1.5)) + rng.normal(0, _PR_JITTER)
            r_times.append(p_times[k] + pr)
            r_from.append(k)
            i += 1
    elif kind == "mobitz2":
        p_drop = rng.uniform(0.08, 0.25)
        params["p_drop"] = p_drop
        prev_blocked = False
        for k in range(n):
            if not prev_blocked and k > 0 and rng.random() < p_drop:
                prev_blocked = True
                continue
            prev_blocked = False
            r_times.append(p_times[k] + pr_base + rng.normal(0, _PR_JITTER))
            r_from.append(k)
    else:                                    # nsr, chf, av1, pavb: 1:1 unless blocked_p says otherwise
        blocked_p = np.zeros(n, bool) if blocked_p is None else blocked_p
        for k in range(n):
            if blocked_p[k]:
                continue
            r_times.append(p_times[k] + pr_base + rng.normal(0, _PR_JITTER))
            r_from.append(k)
    return np.array(r_times), np.array(r_from), params


def generate(kind, n_beats=200, seed=None, n_episodes=2):
    """Generate one synthetic RR series with ``n_beats`` intervals.

    Parameters
    ----------
    kind : one of ``KINDS``.
    n_beats : number of RR intervals returned.
    seed : int or None, for reproducibility.
    n_episodes : number of block episodes (``pavb`` only); each lasts 3 to 6
        blocked P waves.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    rng = np.random.default_rng(seed)
    profile = "chf" if kind == "chf" else "nsr"

    blocked_p, episode_len = None, []
    n_p = int(n_beats * 1.7) + 40 if kind in ("mobitz1", "mobitz2") else n_beats + 40
    if kind == "pavb":
        episode_len = [int(rng.integers(3, 7)) for _ in range(n_episodes)]
        n_p = n_beats + sum(episode_len) + 40
        blocked_p = np.zeros(n_p, bool)
        for e, length in enumerate(episode_len):
            centre = (e + 0.5 + rng.uniform(-0.25, 0.25)) * n_beats / n_episodes
            start = int(np.clip(centre, 15, n_beats - 15))
            blocked_p[start:start + length] = True

    p_times, p = _atrial_times(n_p, rng, profile)
    r_times, r_from, cond = _conduct(p_times, kind, rng, blocked_p)
    r_times, r_from = r_times[:n_beats + 1], r_from[:n_beats + 1]
    if len(r_times) < n_beats + 1:
        raise RuntimeError("not enough conducted beats; increase the P-wave budget")
    return RRSeries(
        kind=kind,
        rr=np.diff(r_times),
        r_times=r_times,
        pause_mask=np.diff(r_from) > 1,
        params={**p, **cond, "episode_len": episode_len},
    )
