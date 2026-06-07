"""Compare les evaluations enregistrees dans l'historique.

Chaque appel a run_eval.py ajoute une ligne dans results/<dataset>/history.jsonl
(configuration + metriques des 3 methodes). Ce script lit cet historique et
affiche des tableaux comparatifs : on voit ainsi l'effet de chaque changement
(stemming, normalisation, alpha, k1/b, modele d'embedding...).

Usage :
    python src/compare_runs.py [--dataset {enron,fiqa}] [--metric NDCG@10]

Sans --metric, affiche les 4 metriques cles (P@10, Recall@10, MRR, NDCG@10).
Genere aussi results/<dataset>/history_comparison.md pour le rapport.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config

METHODS = ["bm25", "semantic", "hybrid"]
METHOD_LABEL = {"bm25": "BM25", "semantic": "Semantique", "hybrid": "Hybride"}
DEFAULT_METRICS = ["P@10", "Recall@10", "MRR", "NDCG@10"]
# Reglages de config affiches pour identifier ce qui change d'un run a l'autre.
CONFIG_KEYS = [
    "bm25_word_normalization", "bm25_remove_stopwords", "bm25_k1", "bm25_b",
    "hybrid_mode", "normalization", "alpha", "embedding_model",
]


def df_to_md(df: pd.DataFrame) -> str:
    """Convertit un DataFrame en tableau Markdown (sans dependance externe)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_history(path) -> list:
    if not path.exists():
        return []
    runs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def runs_overview(runs: list) -> pd.DataFrame:
    """Tableau : un run par ligne, avec son etiquette et sa config cle."""
    rows = []
    for i, r in enumerate(runs):
        row = {"#": i, "label": r["label"], "dossier": r.get("run_dir", ""),
               "date": r["timestamp"], "n_q": r.get("n_queries", ""),
               "qrels": r.get("qrels", "")}
        cfg = r.get("config", {})
        for k in CONFIG_KEYS:
            row[k] = cfg.get(k, "")
        rows.append(row)
    return pd.DataFrame(rows).set_index("#")


def metric_table(runs: list, metric: str) -> pd.DataFrame:
    """Tableau runs x methodes pour une metrique donnee."""
    rows = {}
    for i, r in enumerate(runs):
        key = f"#{i} {r['label']}"
        rows[key] = {
            METHOD_LABEL[m]: round(r["metrics"].get(m, {}).get(metric, 0.0), 4)
            for m in METHODS
        }
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=config.SUPPORTED_DATASETS, default=config.DATASET)
    parser.add_argument("--metric", default=None, help="ex: NDCG@10 (defaut: les 4 cles)")
    args = parser.parse_args()
    config.configure(args.dataset)

    path = config.RESULTS_DIR / "history.jsonl"
    runs = load_history(path)
    if not runs:
        print(f"Aucun historique pour '{config.DATASET}'. Lance d'abord run_eval.py.")
        return

    metrics = [args.metric] if args.metric else DEFAULT_METRICS

    overview = runs_overview(runs)
    print(f"\n=== {len(runs)} run(s) pour '{config.DATASET}' ===\n")
    print(overview.to_string())

    md = [f"# Comparaison des runs ({config.DATASET})\n", "## Configurations\n",
          df_to_md(overview)]
    for metric in metrics:
        tbl = metric_table(runs, metric)
        print(f"\n--- {metric} ---")
        print(tbl.to_string())
        md += [f"\n## {metric}\n", df_to_md(tbl)]

    out = config.RESULTS_DIR / "history_comparison.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n>> Tableau Markdown ecrit : {out}")


if __name__ == "__main__":
    main()
