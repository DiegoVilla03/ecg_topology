import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import tda_ecg as tda


def plot_signal(signal, title="ECG Signal"):
    fig, ax = plt.subplots(figsize=(14, 3), facecolor='white')
    ax.set_facecolor('white')

    ax.plot(np.arange(len(signal)), signal, color='#378ADD', linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Muestra", fontsize=10)
    ax.set_ylabel("Amplitud", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

    plt.tight_layout()
    plt.show()


def plot_takens_2d_scatter(signal, delay, title="Takens 2D — Scatter"):
    embedded = tda.takens_embedding(signal, dimension=2, delay=delay)
    x, y = embedded[:, 0], embedded[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.set_facecolor('white')

    t_color = np.linspace(0, 1, len(x))
    ax.scatter(x, y, c=plt.cm.plasma(t_color), s=1, alpha=0.5)

    ax.set_title(f"{title}  τ={delay}", fontsize=11)
    ax.set_xlabel("x(t)", fontsize=9)
    ax.set_ylabel("x(t + τ)", fontsize=9)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

    plt.tight_layout()
    plt.show()


def plot_takens_2d_lines(signal, delay, title="Takens 2D — Trayectoria"):
    embedded = tda.takens_embedding(signal, dimension=2, delay=delay)
    x, y = embedded[:, 0], embedded[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.set_facecolor('white')

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma',
                        norm=plt.Normalize(0, len(segments)),
                        linewidth=0.5, alpha=0.8)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_title(f"{title}  τ={delay}", fontsize=11)
    ax.set_xlabel("x(t)", fontsize=9)
    ax.set_ylabel("x(t + τ)", fontsize=9)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

    plt.tight_layout()
    plt.show()


def plot_takens_3d_scatter(signal, delay, title="Takens 3D — Scatter"):
    embedded = tda.takens_embedding(signal, dimension=3, delay=delay)
    x, y, z  = embedded[:, 0], embedded[:, 1], embedded[:, 2]

    fig = plt.figure(figsize=(7, 7), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    _style_3d_ax(ax, delay, title)

    t_color = np.linspace(0, 1, len(x))
    ax.scatter(x, y, z, c=plt.cm.plasma(t_color), s=0.3, alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_takens_3d_lines(signal, delay, title="Takens 3D — Trayectoria"):
    embedded = tda.takens_embedding(signal, dimension=3, delay=delay)
    x, y, z  = embedded[:, 0], embedded[:, 1], embedded[:, 2]

    fig = plt.figure(figsize=(7, 7), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    _style_3d_ax(ax, delay, title)

    points   = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap='plasma',
                          norm=plt.Normalize(0, len(segments)),
                          linewidth=0.4, alpha=0.85)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection3d(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    plt.tight_layout()
    plt.show()


def _style_3d_ax(ax, delay, title=""):
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#dddddd')
    ax.grid(False)
    ax.tick_params(colors='black', labelsize=5)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('black')
    ax.set_title(f"{title}  τ={delay}", fontsize=11, pad=6)