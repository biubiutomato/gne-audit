#!/usr/bin/env python3
"""Render all paper figures (Figs. 1-4) at print resolution from stored results.

Reads results/precision_curves.csv, results/metadata.csv and the stored
embeddings in results/*.npy; writes PNGs to figures/ at 400 dpi.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import audit_embeddings as A

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
DPI = 400
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": DPI, "figure.dpi": 100})


def fig1_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.15, 0.7, 1.7, 1.6, "Marker-gene\npanel\n(NCBI)", "#e8f1f8"),
        (2.15, 0.7, 1.8, 1.6, "Representations\nHyenaDNA-tiny\n4-mer / composition", "#e9f6ef"),
        (4.25, 0.7, 1.7, 1.6, "top-k neighbors\ncosine and L2", "#fff3e0"),
        (6.25, 0.7, 1.7, 1.6, "Labels\nfamily / taxon\nlength / GC", "#fdecea"),
        (8.25, 0.7, 1.6, 1.6, "Precision@k\n+ length-\nmatched control", "#f3eef8"),
    ]
    for x, y, w, h, text, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor=c, edgecolor="#333", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.0)
    for x0, x1 in [(1.85, 2.15), (3.95, 4.25), (5.95, 6.25), (7.95, 8.25)]:
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.set_title("GNE-Audit: neighborhood diagnosis for genomic embeddings", fontsize=10, pad=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_protocol.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_fig3():
    c = pd.read_csv(RES / "precision_curves.csv")
    cur = lambda r: c[(c["repr"] == r) & (c.distance == "cosine")].reset_index(drop=True)
    A.plot_precision({"HyenaDNA-tiny / cosine": cur("HyenaDNA-tiny"), "4-mer freq / cosine": cur("4-mer freq")},
                     "Neighborhood precision vs k (cosine)", FIG / "fig2_precision_vs_k.png")
    A.plot_exclusive_stack(cur("HyenaDNA-tiny"), "Exclusive neighbor labels — HyenaDNA-tiny, cosine",
                           FIG / "fig3_stack_HyenaDNA-tiny.png")


def fig4_pca():
    # PCA on l2-normalized vectors (the geometry used by cosine search)
    meta = pd.read_csv(RES / "metadata.csv")
    Z = {name: PCA(2, random_state=0).fit_transform(normalize(np.load(RES / f"{f}.npy")))
         for name, f in [("HyenaDNA-tiny", "HyenaDNA-tiny"), ("4-mer frequency", "4-mer_freq")]}

    # (a) colored by gene family
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, (name, z) in zip(axes, Z.items()):
        for gene, sub in meta.groupby("gene"):
            ax.scatter(z[sub.index, 0], z[sub.index, 1], s=10, alpha=0.7, lw=0, label=gene)
        ax.set_title(name, fontsize=10); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(fontsize=7, frameon=False, loc="upper right", markerscale=1.2)
    fig.suptitle("PCA of embedding neighborhoods colored by gene family", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_pca_family.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (b) colored by GC fraction
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    for ax, (name, z) in zip(axes, Z.items()):
        sc = ax.scatter(z[:, 0], z[:, 1], c=meta["gc"], cmap="coolwarm", vmin=0.35, vmax=0.70, s=8, lw=0)
        ax.set_title(name, fontsize=10); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("PCA of embeddings colored by GC content", fontsize=11)
    fig.tight_layout()
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("GC fraction", fontsize=8)
    fig.savefig(FIG / "fig4_pca_gc.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1_protocol(); fig2_fig3(); fig4_pca()
    print("wrote figures to", FIG)
