"""Synthetic demo of the RR -> Takens -> persistent homology pipeline.

Usage (from the repository root):
    uv run python scripts/run_demo.py

Writes five figures and a results table to ``figures/``.

What this demo can show: the code runs end to end, the embedding and the
diagrams behave as the theory says, and topological features can be compared
against a classical HRV baseline on data whose ground truth is known.
What it cannot show: anything about real patients. See docs/03-synthetic-demo.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ecgtda import persistence, rr_cloud, sliding_window_features, synthetic, tda_features
from ecgtda.hrv import HRV_FEATURES, hrv_features
from ecgtda.topology import TDA_FEATURES

OUT = Path(__file__).resolve().parents[1] / "figures"
SEED = 7
N_BEATS = 200          # RR intervals per window (about 3 minutes)
N_PER_CLASS = 80
MIN_PERSISTENCE = 0.005

LABELS = {"nsr": "Normal sinus", "chf": "Reduced variability (CHF-like)",
          "mobitz1": "AV block II, Wenckebach", "mobitz2": "AV block II, Mobitz II",
          "pavb": "Paroxysmal AV block", "av1": "AV block I"}
COLORS = {"nsr": "#378ADD", "chf": "#E67E22", "mobitz1": "#2E9E5B",
          "mobitz2": "#8E44AD", "pavb": "#C0392B", "av1": "#7F8C8D"}


def style(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------- figures
def fig_rr(examples):
    kinds = ["nsr", "chf", "mobitz1", "mobitz2", "pavb"]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(11, 9), facecolor="white", sharex=True)
    for ax, kind in zip(axes, kinds):
        s = examples[kind] if kind != "pavb" else synthetic.generate("pavb", 200, seed=SEED + 11)
        rr, paused = s.rr[:120], s.pause_mask[:120]
        ax.plot(rr, color=COLORS[kind], lw=1, marker="o", ms=2.5)
        ax.plot(np.flatnonzero(paused), rr[paused], "o", color="black", ms=3.5, label="spans a blocked P wave")
        ax.set_ylabel("RR [s]", fontsize=9)
        ax.set_title(LABELS[kind], fontsize=10, loc="left")
        style(ax)
        if paused.any():
            ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xlabel("Beat", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_01_rr_series.png", dpi=140)
    plt.close(fig)


def fig_embedding(examples):
    kinds = ["nsr", "chf", "mobitz1", "mobitz2"]
    fig = plt.figure(figsize=(11, 9), facecolor="white")
    for i, kind in enumerate(kinds):
        cloud = rr_cloud(examples[kind].rr, delay=1)
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_facecolor("white")
        pts = cloud.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap="plasma", linewidth=0.6, alpha=0.85)
        lc.set_array(np.arange(len(segs)))
        ax.add_collection3d(lc)
        ax.scatter(*cloud.T, c=np.arange(len(cloud)), cmap="plasma", s=5)
        for lim, setter in zip(cloud.T, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            pad = 0.05 * (np.ptp(lim) + 1e-9)
            setter(lim.min() - pad, lim.max() + pad)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#dddddd")
        ax.grid(False)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("RR$_i$", fontsize=8)
        ax.set_ylabel("RR$_{i+1}$", fontsize=8)
        ax.set_zlabel("RR$_{i+2}$", fontsize=8)
        ax.set_title(LABELS[kind] + "  (axes in s, own scale)", fontsize=9)
    fig.suptitle("Delay embedding of the RR series in R$^3$, delay = 1 beat (colour = time)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_02_embedding_3d.png", dpi=140)
    plt.close(fig)


def fig_diagrams(examples):
    kinds = ["nsr", "chf", "mobitz1", "mobitz2"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), facecolor="white")
    for ax, kind in zip(axes, kinds):
        dg = persistence(rr_cloud(examples[kind].rr, delay=1))
        top = max(dg[0][:, 1].max(), dg[1][:, 1].max()) * 1.05
        ax.plot([0, top], [0, top], color="#bbbbbb", lw=0.8)
        ax.scatter(*dg[0].T, s=14, color="#378ADD", label=f"H$_0$ ({len(dg[0])})")
        ax.scatter(*dg[1].T, s=14, color="#C0392B", label=f"H$_1$ ({len(dg[1])})")
        ax.set_xlabel("birth (alpha radius, s)", fontsize=8)
        ax.set_ylabel("death", fontsize=8)
        ax.set_title(LABELS[kind], fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_03_persistence_diagrams.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- classification
def make_dataset(kinds, seed0):
    """kinds: {label: [kind, ...]}; the kinds of a label are cycled."""
    rows = []
    for li, (label, variants) in enumerate(kinds.items()):
        for i in range(N_PER_CLASS):
            kind = variants[i % len(variants)]
            rr = synthetic.generate(kind, N_BEATS, seed=seed0 + 1000 * li + i).rr
            rows.append({"label": label, **hrv_features(rr), **tda_features(rr, 1, MIN_PERSISTENCE)})
    return pd.DataFrame(rows)


def evaluate(df, task):
    feature_sets = {"HRV baseline": HRV_FEATURES, "Topological": TDA_FEATURES,
                    "HRV + topological": HRV_FEATURES + TDA_FEATURES}
    models = {"Random forest": RandomForestClassifier(n_estimators=300, random_state=0),
              "SVM (RBF)": make_pipeline(StandardScaler(), SVC())}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    out = []
    for fs_name, cols in feature_sets.items():
        for m_name, model in models.items():
            r = cross_validate(model, df[cols], df["label"], cv=cv,
                               scoring=["accuracy", "f1_macro"])
            out.append({"task": task, "features": fs_name, "model": m_name,
                        "accuracy": r["test_accuracy"].mean(), "accuracy_sd": r["test_accuracy"].std(),
                        "f1_macro": r["test_f1_macro"].mean()})
    return pd.DataFrame(out)


def to_markdown(df):
    """Minimal markdown table (avoids an extra dependency on tabulate)."""
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def fig_results(res):
    tasks = list(res["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(11, 4), facecolor="white", sharey=True)
    fsets = list(res["features"].unique())
    x = np.arange(len(fsets))
    for ax, task in zip(axes, tasks):
        for j, (m, c) in enumerate(zip(["Random forest", "SVM (RBF)"], ["#378ADD", "#E67E22"])):
            sub = res[(res["task"] == task) & (res["model"] == m)].set_index("features").loc[fsets]
            ax.bar(x + (j - 0.5) * 0.36, sub["accuracy"], 0.36, yerr=sub["accuracy_sd"],
                   color=c, label=m, capsize=3)
        n_classes = 3 if task.startswith("3 classes") else 2
        ax.axhline(1 / n_classes, color="#888888", ls=":", lw=1)
        ax.text(-0.55, 1 / n_classes + 0.015, "chance", fontsize=7, color="#888888", ha="left")
        ax.set_xticks(x, fsets, fontsize=8)
        ax.set_title(task, fontsize=10)
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("Accuracy (5-fold x 3, mean $\\pm$ sd)", fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_04_classification.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- paroxysmal block
def pavb_detection():
    s = synthetic.generate("pavb", 600, seed=SEED + 5, n_episodes=3)
    window, step = 64, 4
    feats = sliding_window_features(s.rr, window=window, step=step, delay=1)
    has_pause = np.array([s.pause_mask[st:st + window].any() for st in feats.index])
    scores = {"H0 max persistence (topological)": feats["h0_max"],
              "H0 max / median persistence (topological)": feats["h0_outlier_ratio"],
              "max RR / median RR (baseline)": feats["max_rr_ratio"]}
    auc = {name: roc_auc_score(has_pause, v) for name, v in scores.items()}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), facecolor="white", sharex=True)
    a1.plot(s.rr, color=COLORS["pavb"], lw=0.9)
    a1.set_ylabel("RR [s]", fontsize=9)
    a1.set_title("Paroxysmal AV block: RR series with three block episodes", fontsize=10, loc="left")
    centres = feats.index.values + window / 2
    a1.plot(np.flatnonzero(s.pause_mask), s.rr[s.pause_mask], "o", color="black", ms=4,
            label="spans a blocked P wave")
    a1.legend(fontsize=8, frameon=False, loc="upper right")
    for c in centres[has_pause]:
        a2.axvspan(c - step / 2, c + step / 2, color="#f3d6d3", alpha=0.7, lw=0)
    a2.text(0.01, 0.96, "shaded: windows that contain a pause", transform=a2.transAxes,
            fontsize=8, va="top", color="#a05a52")
    for ax in (a1, a2):
        style(ax)
    for name, v, c in zip(scores, scores.values(), ["#378ADD", "#8E44AD", "#E67E22"]):
        z = (v - v.min()) / (v.max() - v.min())
        a2.plot(centres, z, label=f"{name}, AUC = {auc[name]:.2f}", color=c, lw=1.4,
                ls="--" if "baseline" in name else "-")
    a2.set_ylim(-0.05, 1.6)
    a2.set_ylabel("score (min-max scaled)", fontsize=9)
    a2.set_xlabel("Beat (window centre)", fontsize=9)
    a2.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_05_sliding_window_pavb.png", dpi=140)
    plt.close(fig)
    return auc


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    examples = {k: synthetic.generate(k, N_BEATS, seed=SEED + i) for i, k in enumerate(synthetic.KINDS)}
    fig_rr(examples)
    fig_embedding(examples)
    fig_diagrams(examples)

    task_a = make_dataset({"normal sinus": ["nsr"], "CHF-like": ["chf"],
                           "AV block II": ["mobitz1", "mobitz2"]}, seed0=10_000)
    task_b = make_dataset({"normal sinus": ["nsr"], "AV block I": ["av1"]}, seed0=50_000)
    res = pd.concat([evaluate(task_a, "3 classes: normal vs CHF-like vs AV block II"),
                     evaluate(task_b, "Negative control: normal vs AV block I")])
    fig_results(res)
    auc = pavb_detection()

    lines = ["# Synthetic demo results", "",
             f"{N_PER_CLASS} synthetic windows per class, {N_BEATS} beats each, seed {SEED}. "
             "Cross-validation: 5 folds x 3 repeats. These numbers describe the generator in "
             "`ecgtda/synthetic.py`, not real patients.", "",
             "## Classification", "",
             to_markdown(res.round(3)), "",
             "## Paroxysmal AV block: window-level detection (AUC)", "",
             "Positive window = contains at least one RR interval that spans a blocked P wave.", "",
             "| score | AUC |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in auc.items()]
    (OUT / "demo_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
