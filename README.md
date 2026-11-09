# GNE-Audit

Code and data for **"Auditing Biological Neighborhoods in Genomic Foundation Model Embeddings"** (Ye Ye, IEEE BIBM 2026 workshop).

GNE-Audit labels the top-*k* neighbors of each query embedding by gene family, genus, phylum, GC similarity and length similarity, and compares family recovery with a length-matched random control. It is applied to 787 bacterial marker-gene sequences embedded with HyenaDNA-tiny and with simple composition baselines (3-mer, 4-mer, GC+length+dinucleotide).

## Quick start

```bash
pip install -r requirements.txt
python audit_embeddings.py     # kNN audit -> results/precision_curves.csv, results/table_k10.csv
python query_examples.py       # 19 query-level failures + AF177295.1 example
python make_figures.py         # Figs. 1-4 of the paper at 400 dpi -> figures/
```

All three steps run on CPU in about a minute. By default `audit_embeddings.py` uses the stored HyenaDNA-tiny embeddings in `results/HyenaDNA-tiny.npy`; pass `--recompute-hyena` to re-embed with the public checkpoint `LongSafari/hyenadna-tiny-1k-seqlen-hf` (needs `torch` and `transformers`; `check_hyena.py` is a quick load test).

## Contents

| Path | What it is |
|---|---|
| `data/markers.jsonl`, `data/markers.fa` | Frozen marker panel: 787 sequences (recA, rpoB, gyrB, dnaK, atpD, 16S rRNA) from 15 genera, centrally windowed to at most 900 bp |
| `fetch_markers.py` | Script used to build the panel from NCBI Nucleotide. NCBI changes over time, so re-running it will not give the identical panel; use `data/` to reproduce the paper |
| `audit_embeddings.py` | Representations, exact kNN search (cosine on l2-normalized vectors, Euclidean on raw vectors), labels, length-matched control |
| `query_examples.py` | Query-level failure count and the illustrative query table |
| `make_figures.py` | Paper figures from the stored results |
| `results/` | Stored embeddings (`*.npy`), `metadata.csv`, `precision_curves.csv`, `table_k10.csv` (Table I), `illustrative_query_AF177295.csv` |
| `figures/` | Paper figures (`fig1`-`fig4`) plus two supplementary plots written by `audit_embeddings.py` |

## Headline numbers (k = 10, cosine)

| Representation | Family | Genus | Phylum | GC | Length | Family, length-matched control |
|---|---|---|---|---|---|---|
| 4-mer freq | 0.715 | 0.610 | 0.871 | 0.641 | 0.750 | 0.178 |
| HyenaDNA-tiny | 0.476 | 0.444 | 0.713 | 0.761 | 0.829 | 0.174 |

Length hit rates are inflated because 541 of 787 sequences are exactly 900 bp after windowing; family claims use the length-matched control.

## Citation

```bibtex
@inproceedings{ye2026gneaudit,
  author    = {Ye Ye},
  title     = {Auditing Biological Neighborhoods in Genomic Foundation Model Embeddings},
  booktitle = {Proc. IEEE Int. Conf. Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```

## License

Code: MIT. Sequences are public NCBI Nucleotide records; accession numbers are in `results/metadata.csv`.
