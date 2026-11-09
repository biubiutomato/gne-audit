#!/usr/bin/env python3
"""Build a multi-gene, multi-taxon bacterial marker-gene panel from NCBI Nucleotide."""
from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

EMAIL = "your.email@example.org"  # NCBI E-utilities asks for a contact address
TOOL = "gne-audit"

GENERA = {
    "Escherichia": "Proteobacteria",
    "Salmonella": "Proteobacteria",
    "Vibrio": "Proteobacteria",
    "Pseudomonas": "Proteobacteria",
    "Helicobacter": "Proteobacteria",
    "Rhizobium": "Proteobacteria",
    "Bacillus": "Firmicutes",
    "Staphylococcus": "Firmicutes",
    "Streptococcus": "Firmicutes",
    "Listeria": "Firmicutes",
    "Clostridioides": "Firmicutes",
    "Mycobacterium": "Actinobacteria",
    "Streptomyces": "Actinobacteria",
    "Corynebacterium": "Actinobacteria",
    "Synechococcus": "Cyanobacteria",
}

GENES = {
    "recA": (600, 1600),
    "rpoB": (600, 4000),
    "gyrB": (600, 3000),
    "dnaK": (600, 2500),
    "atpD": (600, 1800),
    "16S ribosomal RNA": (400, 1800),
}

GENE_LABEL = {
    "recA": "recA",
    "rpoB": "rpoB",
    "gyrB": "gyrB",
    "dnaK": "dnaK",
    "atpD": "atpD",
    "16S ribosomal RNA": "rrs",
}

PER_CELL = 10
SLEEP = 0.34


def eutils(endpoint: str, **params) -> bytes:
    params = dict(params)
    params.update(email=EMAIL, tool=TOOL)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}.fcgi?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL}/0.1"})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed {endpoint}: {last}")


def search_ids(gene: str, genus: str, lo: int, hi: int, retmax: int = 30) -> list[str]:
    if gene.startswith("16S"):
        term = f'{genus}[Organism] AND 16S ribosomal RNA[Title] AND {lo}:{hi}[SLEN]'
    else:
        term = (
            f'{genus}[Organism] AND {gene}[Gene Name] AND {lo}:{hi}[SLEN] '
            f'AND ("complete cds"[Title] OR "partial cds"[Title] OR CDS[Feature key])'
        )
    raw = eutils("esearch", db="nuccore", term=term, retmax=str(retmax), retmode="json")
    data = json.loads(raw.decode())
    ids = data.get("esearchresult", {}).get("idlist", [])
    if ids:
        return ids
    # fallback looser
    if gene.startswith("16S"):
        return []
    term2 = f'{genus}[Organism] AND {gene}[All Fields] AND {lo}:{hi}[SLEN] NOT shotgun[Title]'
    raw = eutils("esearch", db="nuccore", term=term2, retmax=str(retmax), retmode="json")
    data = json.loads(raw.decode())
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_fasta(ids: list[str]) -> str:
    if not ids:
        return ""
    raw = eutils("efetch", db="nuccore", id=",".join(ids), rettype="fasta", retmode="text")
    return raw.decode("utf-8", errors="replace")


def parse_fasta(text: str) -> list[dict]:
    recs = []
    acc = header = None
    seq_chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc and seq_chunks:
                recs.append({"acc": acc, "header": header, "seq": "".join(seq_chunks)})
            header = line[1:].strip()
            acc = header.split()[0]
            seq_chunks = []
        else:
            seq_chunks.append(line.strip())
    if acc and seq_chunks:
        recs.append({"acc": acc, "header": header, "seq": "".join(seq_chunks)})
    return recs


ORG_RE = re.compile(r"\[([^\]]+)\]\s*$")


def organism_from_header(header: str) -> str:
    m = ORG_RE.search(header)
    if m:
        return m.group(1)
    parts = header.split()
    if len(parts) >= 3:
        return " ".join(parts[1:3])
    return "unknown"


def clean_seq(seq: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", seq).upper()


def window(seq: str, L: int = 900) -> str:
    if len(seq) <= L:
        return seq
    start = (len(seq) - L) // 2
    return seq[start : start + L]


def main():
    rng = random.Random(42)
    out_json = OUT / "markers.jsonl"
    # resume
    seen_seq = set()
    seen_cell = {}
    n_exist = 0
    if out_json.exists():
        with out_json.open() as f:
            for line in f:
                r = json.loads(line)
                key = r["seq"][:240] + r["seq"][-60:]
                seen_seq.add(key)
                cell = (r["gene"], r["genus"])
                seen_cell[cell] = seen_cell.get(cell, 0) + 1
                n_exist += 1
        print(f"resuming with {n_exist} rows")

    with out_json.open("a") as fout:
        for gene, (lo, hi) in GENES.items():
            label = GENE_LABEL[gene]
            for genus, phylum in GENERA.items():
                already = seen_cell.get((label, genus), 0)
                if already >= PER_CELL:
                    print(f"skip {label}/{genus} ({already})")
                    continue
                print(f"{label} / {genus}", flush=True)
                try:
                    ids = search_ids(gene, genus, lo, hi)
                except Exception as exc:
                    print("  search failed", exc)
                    time.sleep(1)
                    continue
                time.sleep(SLEEP)
                if not ids:
                    print("  no ids")
                    continue
                rng.shuffle(ids)
                ids = ids[:28]
                try:
                    fasta = fetch_fasta(ids)
                except Exception as exc:
                    print("  fetch failed", exc)
                    continue
                time.sleep(SLEEP)
                kept = already
                new = 0
                for rec in parse_fasta(fasta):
                    seq = window(clean_seq(rec["seq"]), 900)
                    if len(seq) < 280:
                        continue
                    key = seq[:240] + seq[-60:]
                    if key in seen_seq:
                        continue
                    seen_seq.add(key)
                    org = organism_from_header(rec["header"])
                    gc = (seq.count("G") + seq.count("C")) / len(seq)
                    row = {
                        "acc": rec["acc"],
                        "gene": label,
                        "genus": genus,
                        "phylum": phylum,
                        "organism": org,
                        "length": len(seq),
                        "gc": gc,
                        "header": rec["header"],
                        "seq": seq,
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    kept += 1
                    new += 1
                    seen_cell[(label, genus)] = kept
                    if kept >= PER_CELL:
                        break
                print(f"  +{new} (total cell {kept})")
    # fasta snapshot
    rows = [json.loads(l) for l in out_json.read_text().splitlines() if l.strip()]
    fa = OUT / "markers.fa"
    with fa.open("w") as f:
        for i, r in enumerate(rows):
            f.write(f">{i}|{r['acc']}|{r['gene']}|{r['genus']}\n")
            f.write(r["seq"] + "\n")
    print(f"wrote {len(rows)} records")


if __name__ == "__main__":
    main()
