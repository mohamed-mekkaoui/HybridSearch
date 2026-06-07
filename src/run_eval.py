"""Script : evalue les trois methodes et genere le rapport + les figures.

Usage :
    python src/run_eval.py [--dataset {enron,fiqa}] [--label "ma config"] [--max-queries N]

Produit dans results/<dataset>/<label>/ (un dossier par run) :
    - metrics.json            : metriques de ce run
    - comparison.md           : tableau Markdown (pour le rapport)
    - comparison_metrics.png  : graphique des metriques
    - comparison_time.png     : graphique des temps
    - embeddings_projection.png : projection 2D des embeddings
Et dans results/<dataset>/ :
    - history.jsonl           : HISTORIQUE de tous les runs (config + metriques),
                                pour comparer les experiences (cf. compare_runs.py)
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import config
import evaluation
import indexing
import visualization
from search import SearchEngine


def safe_name(label: str) -> str:
    """Transforme une etiquette en nom de dossier sur (alphanum, -, _, .)."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return name or "run"


def append_history(summary: dict, label: str, n_queries: int, manual: bool,
                   run_dir_name: str) -> None:
    """Ajoute une ligne a results/<dataset>/history.jsonl (un run = une ligne JSON)."""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label or config.auto_label(),
        "run_dir": run_dir_name,
        "dataset": config.DATASET,
        "n_queries": n_queries,
        "qrels": "manuel/officiel" if manual else "pooling",
        "config": config.snapshot(),
        "metrics": summary,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_markdown_table(summary: dict) -> str:
    header = "| Methode | P@10 | Recall@10 | MRR | NDCG@10 | Temps moyen (ms) |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    label = {"bm25": "BM25", "semantic": "Semantique", "hybrid": "Hybride"}
    for method, row in summary.items():
        lines.append(
            f"| {label.get(method, method)} "
            f"| {row.get('P@10', 0):.3f} "
            f"| {row.get('Recall@10', 0):.3f} "
            f"| {row.get('MRR', 0):.3f} "
            f"| {row.get('NDCG@10', 0):.3f} "
            f"| {row.get('time_mean_ms', 0):.1f} |"
        )
    return "\n".join(lines)


def main(label: str = None, max_queries: int = None):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f">> Dataset actif : {config.DATASET}")

    print(">> Chargement du moteur de recherche...")
    engine = SearchEngine.from_artifacts()

    print(">> Chargement des requetes...")
    queries_raw = evaluation.load_queries()
    queries = [item["query"] for item in queries_raw]

    # Si aucun jugement manuel n'est fourni, on genere des pseudo-qrels.
    has_manual = any(item.get("relevant") for item in queries_raw)
    if has_manual:
        print("   Jugements manuels detectes dans queries.json.")
        queries_with_rels = queries_raw
    else:
        print(">> Generation des pseudo-qrels par pooling (semi-automatique)...")
        qrels = evaluation.build_pseudo_qrels(engine, queries)
        with open(config.QRELS_JSON, "w", encoding="utf-8") as f:
            json.dump(qrels, f, indent=2)
        queries_with_rels = [
            {"query": q, "relevant": qrels[q]} for q in queries
        ]
        n_rel = sum(len(v) for v in qrels.values())
        print(f"   {n_rel} jugements de pertinence generes.")

    if max_queries is not None and max_queries < len(queries_with_rels):
        queries_with_rels = queries_with_rels[:max_queries]
        print(f"   Limite a {max_queries} requetes (--max-queries).")

    print(">> Evaluation des 3 methodes...")
    summary = evaluation.evaluate(engine, queries_with_rels)
    n_eval = sum(1 for q in queries_with_rels if q.get("relevant"))

    # Dossier dedie a ce run : results/<dataset>/<label>/ (figures + metriques)
    run_name = safe_name(label) if label else "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegardes dans le dossier du run
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = to_markdown_table(summary)
    with open(run_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write(f"# Resultats ({config.DATASET} / {run_name})\n\n" + table + "\n")
    print("\n" + table + "\n")

    print(">> Generation des graphiques...")
    visualization.plot_metric_comparison(summary, save_path=run_dir / "comparison_metrics.png")
    visualization.plot_time_comparison(summary, save_path=run_dir / "comparison_time.png")
    embeddings = indexing.load_embeddings()
    visualization.plot_embeddings(embeddings, save_path=run_dir / "embeddings_projection.png")

    # Historique global (toutes les executions) a la racine du dataset.
    append_history(summary, label, n_eval, has_manual, run_name)

    print(f">> Termine. Resultats de ce run dans {run_dir}")
    print(f">> Run '{label or config.auto_label()}' ajoute a l'historique. "
          "Comparer : python src/compare_runs.py --dataset " + config.DATASET)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=config.SUPPORTED_DATASETS, default=config.DATASET,
        help="dataset a evaluer (defaut: enron)",
    )
    parser.add_argument(
        "--label", default=None,
        help="etiquette du run pour l'historique (ex: 'baseline', 'avec stemming')",
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="limiter le nombre de requetes evaluees (defaut: toutes)",
    )
    args = parser.parse_args()
    config.configure(args.dataset)
    main(label=args.label, max_queries=args.max_queries)
