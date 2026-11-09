#!/usr/bin/env python3
"""Query-level failure cases for HyenaDNA-tiny (cosine, k=10).

Counts queries with no same-family neighbor among the top 10 while at least
8 of the 10 neighbors are GC-matched (|dGC| < 0.02), and writes the
neighborhood of the illustrative query AF177295.1 used in the paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
K, GC_TOL, QUERY = 10, 0.02, "AF177295.1"

meta = pd.read_csv(RES / "metadata.csv")
X = normalize(np.load(RES / "HyenaDNA-tiny.npy"))
nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute").fit(X)
_, idx = nn.kneighbors(X)

gene, genus, gc = meta["gene"].to_numpy(), meta["genus"].to_numpy(), meta["gc"].to_numpy()
fails = []
for q in range(len(meta)):
    nb = [j for j in idx[q] if j != q][:K]
    fam = sum(gene[j] == gene[q] for j in nb)
    gcm = sum(abs(gc[j] - gc[q]) < GC_TOL for j in nb)
    if fam == 0 and gcm >= 8:
        fails.append((meta.acc[q], gene[q], genus[q], round(gc[q], 3), gcm))
print(f"{len(fails)} of {len(meta)} queries: 0 same-family neighbors and >=8 GC-matched in top {K}")
for f in fails:
    print("  ", *f)

q = int(np.flatnonzero(meta["acc"].to_numpy() == QUERY)[0])
nb = [j for j in idx[q] if j != q][:K]
rows = [dict(role="query", rank=0, **meta.iloc[q].to_dict())]
rows += [dict(role="neighbor", rank=r, **meta.iloc[j].to_dict()) for r, j in enumerate(nb, 1)]
out = pd.DataFrame(rows)
out["gc_diff_from_query"] = (out["gc"] - gc[q]).abs()
out["same_gene"] = out["gene"] == gene[q]
out["same_genus"] = out["genus"] == genus[q]
out.to_csv(RES / f"illustrative_query_{QUERY.split('.')[0]}.csv", index=False)
same_genus = out[(out.role == "neighbor") & out.same_genus]
print(f"{QUERY}: same family {int(out.same_gene[1:].sum())}/10, GC-matched "
      f"{int((out.gc_diff_from_query[1:] < GC_TOL).sum())}/10, same genus {len(same_genus)}/10, "
      f"same-genus genes: {same_genus.gene.value_counts().to_dict()}")
