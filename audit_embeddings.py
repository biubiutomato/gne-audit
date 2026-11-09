#!/usr/bin/env python3
"""Neighborhood audit of genomic embeddings (CPU-feasible representations)."""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "markers.jsonl"
OUT = ROOT
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)

K_GRID = [1, 5, 10, 20, 50]
RNG = np.random.default_rng(42)
BASES = "ACGT"


def load_records():
    recs = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    # drop near-duplicate full sequences
    seen = set()
    uniq = []
    for r in recs:
        k = r["seq"]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq


def kmer_embed(seqs, k=4):
    vocab = ["".join(p) for p in itertools.product(BASES, repeat=k)]
    idx = {mer: i for i, mer in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) < k:
            continue
        counts = np.zeros(len(vocab), dtype=np.float32)
        for j in range(len(s) - k + 1):
            mer = s[j : j + k]
            t = idx.get(mer)
            if t is not None:
                counts[t] += 1
        tot = counts.sum()
        if tot > 0:
            X[i] = counts / tot
    return X


def composition_embed(seqs, lengths, gcs):
    """GC, log-length, and 16 dinucleotide frequencies."""
    idx = {a + b: i for i, (a, b) in enumerate(itertools.product(BASES, repeat=2))}
    X = np.zeros((len(seqs), 18), dtype=np.float32)
    for i, s in enumerate(seqs):
        din = np.zeros(16, dtype=np.float32)
        for j in range(len(s) - 1):
            mer = s[j : j + 2]
            t = idx.get(mer)
            if t is not None:
                din[t] += 1
        if din.sum() > 0:
            din /= din.sum()
        X[i, 0] = gcs[i]
        X[i, 1] = math.log(max(lengths[i], 1))
        X[i, 2:] = din
    return X


def try_hyena_embed(seqs, max_len=1022):
    """Load HyenaDNA-tiny if transformers + weights fit in memory."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        print("transformers unavailable", exc)
        return None
    name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    print("loading", name)
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
        model.eval()
    except Exception as exc:
        print("hyena load failed", exc)
        return None
    embs = []
    bs = 2
    with torch.inference_mode():
        for i in range(0, len(seqs), bs):
            batch = [s[:max_len] for s in seqs[i : i + bs]]
            try:
                inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
                out = model(**inputs)
                hidden = out.last_hidden_state  # [B, L, D]
                mask = inputs.get("attention_mask")
                if mask is not None:
                    m = mask.unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    vec = hidden.mean(1)
                embs.append(vec.cpu().numpy())
            except Exception as exc:
                print("batch failed", i, exc)
                d = embs[0].shape[1] if embs else 128
                embs.append(np.zeros((len(batch), d), dtype=np.float32))
            if i % 50 == 0:
                print(f"  hyena {i}/{len(seqs)}")
    return np.concatenate(embs, axis=0).astype(np.float32)


def l2_normalize(X):
    return normalize(X, norm="l2")


def pairwise_labels(recs):
    n = len(recs)
    gene = np.array([r["gene"] for r in recs])
    genus = np.array([r["genus"] for r in recs])
    phylum = np.array([r["phylum"] for r in recs])
    length = np.array([r["length"] for r in recs], dtype=np.float32)
    gc = np.array([r["gc"] for r in recs], dtype=np.float32)
    return gene, genus, phylum, length, gc


def neighbor_audit(X, recs, metric, max_k=50):
    n = len(recs)
    gene, genus, phylum, length, gc = pairwise_labels(recs)
    nn = NearestNeighbors(n_neighbors=min(max_k + 1, n), metric=metric, algorithm="brute")
    nn.fit(X)
    dist, ind = nn.kneighbors(X, return_distance=True)
    # drop self (first neighbor if identical)
    neigh = []
    for i in range(n):
        ids = [j for j in ind[i] if j != i][:max_k]
        neigh.append(ids)
    return neigh, gene, genus, phylum, length, gc


def precision_curves(neigh, recs, gene, genus, phylum, length, gc, ks=K_GRID):
    n = len(recs)
    rows = []
    # matched controls: for each query, sample k items in same length bin, not self
    length_bins = np.digitize(length, np.quantile(length, np.linspace(0, 1, 11)[1:-1]))
    gc_bins = np.digitize(gc, np.quantile(gc, np.linspace(0, 1, 11)[1:-1]))
    by_lenbin = defaultdict(list)
    by_gcbin = defaultdict(list)
    for i in range(n):
        by_lenbin[int(length_bins[i])].append(i)
        by_gcbin[int(gc_bins[i])].append(i)

    for k in ks:
        fam = tax = phy = len_only = gc_only = both_art = none = 0
        total = 0
        raw = dict(family=0, genus=0, phylum=0, length=0, gc=0)
        ctrl_fam = ctrl_tax = 0
        ctrl_tot = 0
        for i in range(n):
            ids = neigh[i][:k]
            if not ids:
                continue
            for j in ids:
                same_fam = gene[i] == gene[j]
                same_gen = genus[i] == genus[j]
                same_phy = phylum[i] == phylum[j]
                rel_len = abs(length[i] - length[j]) / max(length[i], 1)
                rel_gc = abs(gc[i] - gc[j])
                len_sim = rel_len < 0.10
                gc_sim = rel_gc < 0.02
                raw["family"] += int(same_fam)
                raw["genus"] += int(same_gen)
                raw["phylum"] += int(same_phy)
                raw["length"] += int(len_sim)
                raw["gc"] += int(gc_sim)
                # exclusive hierarchy
                if same_fam:
                    fam += 1
                elif same_gen:
                    tax += 1
                elif same_phy:
                    phy += 1
                elif len_sim and gc_sim:
                    both_art += 1
                elif len_sim:
                    len_only += 1
                elif gc_sim:
                    gc_only += 1
                else:
                    none += 1
                total += 1
            # length-matched control
            pool = [t for t in by_lenbin[int(length_bins[i])] if t != i]
            if pool:
                take = RNG.choice(pool, size=min(k, len(pool)), replace=False)
                for j in take:
                    ctrl_fam += int(gene[i] == gene[j])
                    ctrl_tax += int(genus[i] == genus[j])
                    ctrl_tot += 1
        if total == 0:
            continue
        rows.append(
            {
                "k": k,
                "n_pairs": total,
                "excl_family": fam / total,
                "excl_genus": tax / total,
                "excl_phylum": phy / total,
                "excl_len_gc": both_art / total,
                "excl_len_only": len_only / total,
                "excl_gc_only": gc_only / total,
                "excl_other": none / total,
                "hit_family": raw["family"] / total,
                "hit_genus": raw["genus"] / total,
                "hit_phylum": raw["phylum"] / total,
                "hit_length": raw["length"] / total,
                "hit_gc": raw["gc"] / total,
                "ctrl_family": (ctrl_fam / ctrl_tot) if ctrl_tot else float("nan"),
                "ctrl_genus": (ctrl_tax / ctrl_tot) if ctrl_tot else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_precision(df_map, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=160)
    colors = {
        "hit_family": "#1b6ca8",
        "hit_genus": "#2a9d8f",
        "hit_phylum": "#7b6bb3",
        "hit_gc": "#e09f3e",
        "hit_length": "#d1495b",
        "ctrl_family": "#1b6ca8",
    }
    for ax, (name, df) in zip(axes, df_map.items()):
        for col, lab, ls in [
            ("hit_family", "same gene family", "-"),
            ("hit_genus", "same genus", "-"),
            ("hit_phylum", "same phylum", "-"),
            ("hit_gc", "GC-similar (|Δ|<0.02)", "--"),
            ("hit_length", "length-similar (|Δ|<10%)", "--"),
            ("ctrl_family", "family, length-matched ctrl", ":"),
        ]:
            ax.plot(df["k"], df[col], ls=ls, marker="o", ms=4, lw=1.8, color=colors[col], label=lab)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("Precision@k")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(K_GRID)
        ax.grid(True, alpha=0.3, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_exclusive_stack(df, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.5), dpi=160)
    ks = df["k"].to_numpy()
    cols = [
        ("excl_family", "gene family", "#1b6ca8"),
        ("excl_genus", "genus (not family)", "#2a9d8f"),
        ("excl_phylum", "phylum only", "#7b6bb3"),
        ("excl_len_gc", "length+GC only", "#9c6644"),
        ("excl_len_only", "length only", "#d1495b"),
        ("excl_gc_only", "GC only", "#e09f3e"),
        ("excl_other", "other / unknown", "#b8b8b8"),
    ]
    bottom = np.zeros(len(df))
    x = np.arange(len(ks))
    for key, lab, col in cols:
        vals = df[key].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=col, width=0.7, label=lab)
        bottom += vals
    ax.set_xticks(x, ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Exclusive label share")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_vs_l2(table, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    metrics = ["hit_family", "hit_genus", "hit_phylum", "hit_gc", "hit_length"]
    labels = ["family", "genus", "phylum", "GC", "length"]
    x = np.arange(len(metrics))
    width = 0.35
    # expect columns model_metric
    models = []
    for name in table["repr"].unique():
        models.append(name)
    # plot first two rows at k=10 cosine vs l2 of first FM-like repr
    sub = table[(table["k"] == 10)]
    grouped = list(sub.groupby("repr"))
    for i, (name, g) in enumerate(grouped[:2]):
        vals = [g[m].mean() for m in metrics]
        ax.bar(x + (i - 0.5) * width, vals, width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hit rate @ k=10")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    recs = load_records()
    print("n sequences", len(recs))
    if len(recs) < 40:
        raise SystemExit("not enough sequences yet")
    seqs = [r["seq"] for r in recs]
    lengths = np.array([r["length"] for r in recs], dtype=np.float32)
    gcs = np.array([r["gc"] for r in recs], dtype=np.float32)
    meta = pd.DataFrame(
        [{k: r[k] for k in ("acc", "gene", "genus", "phylum", "organism", "length", "gc")} for r in recs]
    )
    meta.to_csv(RES / "metadata.csv", index=False)
    print(meta.groupby(["gene", "phylum"]).size().unstack(fill_value=0))

    reprs = {}
    X4 = kmer_embed(seqs, 4)
    reprs["4-mer freq"] = X4
    X3 = kmer_embed(seqs, 3)
    reprs["3-mer freq"] = X3
    XC = composition_embed(seqs, lengths, gcs)
    reprs["GC+length+dinuc"] = XC

    cached = RES / "HyenaDNA-tiny.npy"
    if "--recompute-hyena" not in sys.argv and cached.exists():
        print("using cached HyenaDNA-tiny embeddings", cached)
        Xh = np.load(cached)
    else:
        Xh = try_hyena_embed(seqs)
    if Xh is not None and Xh.shape[0] == len(seqs):
        reprs["HyenaDNA-tiny"] = Xh
        np.save(RES / "hyena_tiny.npy", Xh)

    for name, X in reprs.items():
        np.save(RES / (name.replace(" ", "_").replace("+", "_") + ".npy"), X)

    all_rows = []
    curve_store = {}
    for name, X in reprs.items():
        Xn = l2_normalize(X)
        for metric, tag in [("cosine", "cosine"), ("euclidean", "L2")]:
            neigh, gene, genus, phylum, length, gc = neighbor_audit(Xn if metric == "cosine" else X, recs, metric)
            df = precision_curves(neigh, recs, gene, genus, phylum, length, gc)
            df["repr"] = name
            df["distance"] = tag
            all_rows.append(df)
            key = f"{name} / {tag}"
            curve_store[key] = df
            print(name, tag)
            print(df[["k", "hit_family", "hit_genus", "hit_gc", "hit_length", "ctrl_family"]].to_string(index=False))

    tab = pd.concat(all_rows, ignore_index=True)
    tab.to_csv(RES / "precision_curves.csv", index=False)

    # main figure: best biological vs composition, cosine
    pick = []
    for name in ["HyenaDNA-tiny", "4-mer freq", "GC+length+dinuc"]:
        key = f"{name} / cosine"
        if key in curve_store:
            pick.append((key, curve_store[key]))
    if len(pick) >= 2:
        plot_precision(dict(pick[:2]), "Neighborhood precision vs k (cosine)", FIG / "fig2_precision_vs_k.png")
    # exclusive stack for 4-mer cosine and hyena if present
    for name in ["HyenaDNA-tiny", "4-mer freq"]:
        key = f"{name} / cosine"
        if key in curve_store:
            safe = name.replace(" ", "_")
            plot_exclusive_stack(curve_store[key], f"Exclusive neighbor labels — {name}, cosine", FIG / f"fig3_stack_{safe}.png")

    # cosine vs l2 table plot at k=10
    plot_cosine_vs_l2(tab, FIG / "fig3_cosine_vs_l2.png")

    # summary table k=10
    sub = tab[tab["k"] == 10].copy()
    cols = [
        "repr",
        "distance",
        "hit_family",
        "hit_genus",
        "hit_phylum",
        "hit_gc",
        "hit_length",
        "ctrl_family",
        "ctrl_genus",
    ]
    sub[cols].to_csv(RES / "table_k10.csv", index=False)
    print("\n=== k=10 summary ===")
    print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("wrote", RES, FIG)


if __name__ == "__main__":
    main()
