# HOD-Mamba

Code and notebooks for the paper

> **HOD-Mamba: Scalable, Hardware-Constrained Analytics for Heterogeneous Spatio-Temporal Discourse Graphs — From Dialect-Aware Semantic Fusion to Tree-Recurrent Opinion Forecasting and Modularity-Aware Community Detection**
> Noureddine Azzouza and Karima Akli-Astouati. Submitted to *The VLDB Journal*, special issue on Spatio-Temporal Data Management and Analytics (2026).

HOD-Mamba models a discussion forum as a heterogeneous spatio-temporal graph (users, documents, reply trees, timestamps). It provides:

- **Tree-recurrent opinion forecasting.** A selective state-space model whose recurrence follows each reply's *parent* instead of the previous message in time. A dual variant also reads the chronological history.
- **Parallel and incremental execution.** The recurrence is affine in its state, so it runs level-synchronously or by pointer jumping (logarithmic depth), with results identical to the sequential loop. Its per-document states form an append-only store that updates a forecast in about 1.7 ms per new reply.
- **Community detection.** A graph-aware modularity head (DMoN) on the learned author embeddings, compared with K-means, spectral clustering, METIS, Louvain and Leiden, plus month-by-month community dynamics.

Every experiment runs on a **single free-tier NVIDIA T4 (16 GB)** on Google Colab or Kaggle.

---

## Repository layout

```
.
├── notebooks/
│   ├── 01_reddit_collection.ipynb          # data collection (Reddit API)
│   ├── 01b_raw_data_enrichment.ipynb       # reply-tree links, temporal and co-participation edges, author features
│   ├── 03_labeling_frozen.ipynb            # DziriBERT embeddings + frozen-encoder classifiers
│   ├── 03b_finetune_sentiment_emotion.ipynb# fine-tuned DziriBERT labeler (nested 5-fold CV)
│   ├── 04_dataset_construction.ipynb       # hyperedges, temporal 70/15/15 split, author graph
│   ├── 05_hod_mamba_lite.ipynb             # HOD-Mamba-Lite
│   ├── 06_hod_mamba_hetero.ipynb           # HOD-Mamba-Hetero + DMoN (original version)
│   ├── 07_ablation_tree_vs_chain.ipynb     # first tree-vs-chain ablation
│   ├── 08_gru_baseline_k_sweep.ipynb       # GRU baseline, first K sweep
│   ├── 09_forecasting_baselines_v4.ipynb   # FINAL forecasting results (Tables 4-5, leakage audit)
│   ├── 10_parallel_tree_scan_scalability_v2.ipynb  # schedules, scaling, streaming, Table 3
│   └── 11_community_detection.ipynb        # FINAL community results, K sweep, dynamics
├── hod_core.py                             # shared models, scans and data loading
├── requirements.txt
├── LICENSE
└── README.md
```

**Which notebooks produce the paper's numbers.** Notebooks 05–08 are kept for transparency, but the reported results come from **09 (v4), 10 and 11**. Notebooks 05–08 used an author feature that leaked the target label (an author's average sentiment over the whole corpus). Notebook 09 documents and quantifies that leak, and all final results use no author features (paper, Section 7.3).

---

## Data

The derived dataset is on Zenodo: **https://doi.org/10.5281/zenodo.23300314**

It contains document IDs, thread structure, timestamps, pseudonymized author IDs, sentiment and emotion labels (human consensus labels for the 916-document gold set, model-generated labels elsewhere), frozen DziriBERT embeddings, the hyperedge splits and the author graph.

Unpack the Zenodo archive so that it matches this layout:

```
<BASE_DIR>/
├── processed_v2/
│   ├── dataset/
│   │   ├── dataset_contract.json
│   │   ├── vocab.json
│   │   ├── train_items.pkl  val_items.pkl  test_items.pkl
│   │   ├── node_feat_matrix.npy
│   │   └── static_author_graph.parquet
│   ├── dziribert_embeddings.npy            # 210,994 x 768
│   ├── text_corpus.parquet                 # metadata only (no text column)
│   ├── text_corpus_labeled_finetuned.parquet
│   ├── temporal_edges.parquet
│   └── finetune_contract.json  labeling_contract.json
└── annotation/
    └── gold_labels_consensus.parquet       # doc_id, language_tag, sentiment, emotion (no text)
```

`dataset_contract.json` stores absolute paths. If your `BASE_DIR` differs from the original, edit the paths in that file, or recreate it by running notebook 04.

---

## Running the experiments

### On Google Colab

1. Upload the data to your Google Drive (layout above).
2. Open a notebook in Colab and select **Runtime → Change runtime type → T4 GPU**.
3. In the first configuration cell, set `BASE_DIR`, e.g. `"/content/drive/MyDrive/PhD/reddit_opinion_research"`.
4. Run all cells. Results are written to `<BASE_DIR>/processed_v2/results/<notebook>/`.


### Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

A CUDA GPU with at least 10 GB is recommended. The Dual model with the pointer-jumping schedule peaks at 9.8 GB; the level-synchronous schedule needs about 2.7 GB.

### Smoke test, no data needed

Notebooks 09–11 can run end to end on a small synthetic dataset with the same schema, on CPU:

```bash
HOD_SMOKE=1 jupyter nbconvert --to notebook --execute notebooks/09_forecasting_baselines_v4.ipynb
```

### Order and runtime (T4)

| Notebook | Produces | Approx. time |
|---|---|---|
| 09 v4 | forecasting table (14 forecasters × 5 seeds), bootstrap contrasts, depth/branching analysis, leakage audit, gold-subset check | ~1.5 h (resumable) |
| 10 v2 | exactness of the three schedules, depth statistics, scan microbenchmarks, epoch time and memory, scaling, streaming replay, Table 3 | ~1 h |
| 11 | community detection (Hetero and Lite × 5 seeds), graph baselines, K sweep, community profiles, monthly dynamics | ~1 h (resumable) |

Notebooks 09 and 11 save each (model, seed) as soon as it finishes; after a disconnect, re-run all cells and they continue where they stopped.

---

## Main results (from the paper)

| Result | Value |
|---|---|
| Tree vs chronological conditioning, sentiment macro-F1 (SSM / GRU / Transformer) | +0.012 / +0.016 / +0.017, all bootstrap CIs exclude 0 |
| HOD-Mamba-Dual vs persistence baseline (sentiment / emotion) | +0.021 / +0.011, both significant |
| Training epoch speed-up over the sequential scan (Lite, pointer jumping) | 3.65× |
| Incremental update per new reply (median / p99) | 1.74 ms / 3.14 ms |
| DMoN vs K-means modularity at K = 12 | 6.4× |
| Association of communities with members' sentiment, DMoN vs METIS (Cramér's V) | 0.154 vs 0.055 |

---

## Citation

```bibtex
@article{azzouza2026hodmamba,
  title   = {{HOD-Mamba}: Scalable, Hardware-Constrained Analytics for Heterogeneous
             Spatio-Temporal Discourse Graphs},
  author  = {Azzouza, Noureddine and Akli-Astouati, Karima},
  journal = {The VLDB Journal},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the dataset (Zenodo DOI above) if you use it.

## License

Code: MIT (see `LICENSE`). Derived dataset: CC BY 4.0 (see the Zenodo record). Reddit content remains subject to Reddit's terms.

## Contact

Noureddine Azzouza — RIIMA Laboratory - USTHB, Bab ezzouar, Algiers, Algeria — azzouza.n@gmail.com
