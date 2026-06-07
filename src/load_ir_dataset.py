"""Chargeur d'un benchmark BEIR (FiQA) au format du projet.

FiQA est un dataset de Recherche d'Information du domaine financier, livre avec
ses jugements de pertinence officiels (qrels). On le telecharge via la librairie
`ir_datasets`, puis on le convertit dans EXACTEMENT le meme format que notre
corpus Enron :
  - artifacts/fiqa/corpus.parquet : colonnes doc_id (entier), subject, body, text
  - data/queries_fiqa.json        : [{"query": ..., "relevant": [ids entiers]}]

Les doc_id texte de BEIR sont remappes en entiers 0..N-1 (le schema attendu par
evaluation.py). Pour limiter le temps d'embedding, le corpus est plafonne a
config.FIQA_MAX_DOCS, mais TOUS les documents references dans les qrels sont
conserves (sinon le Recall serait fausse) ; les autres documents (distracteurs)
sont echantillonnes par reservoir sampling deterministe.

Usage :
    python src/load_ir_dataset.py
"""
from __future__ import annotations

import json
import random

import pandas as pd

import config

config.configure("fiqa")


def _doc_text(doc) -> str:
    """Concatene titre (si present) et texte d'un document BEIR."""
    title = getattr(doc, "title", "") or ""
    text = getattr(doc, "text", "") or ""
    return (title + ". " + text).strip(". ").strip() if title else text.strip()


def load_fiqa() -> tuple[pd.DataFrame, list[dict]]:
    import ir_datasets

    # ---- qrels + requetes (split test) ---------------------------------
    test = ir_datasets.load(config.FIQA_QRELS_DATASET)
    query_text = {q.query_id: q.text for q in test.queries_iter()}

    qrels: dict[str, list[str]] = {}
    judged_ids: set[str] = set()
    for qrel in test.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)
            judged_ids.add(qrel.doc_id)

    # ---- documents (corpus complet) ------------------------------------
    docs = ir_datasets.load(config.FIQA_DOCS_DATASET)
    budget = max(0, config.FIQA_MAX_DOCS - len(judged_ids))
    rng = random.Random(config.RANDOM_SEED)

    kept: list[tuple[str, str]] = []        # docs juges (toujours conserves)
    reservoir: list[tuple[str, str]] = []   # distracteurs echantillonnes
    seen_nonjudged = 0

    for doc in docs.docs_iter():
        item = (doc.doc_id, _doc_text(doc))
        if doc.doc_id in judged_ids:
            kept.append(item)
        elif budget > 0:
            if len(reservoir) < budget:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen_nonjudged)
                if j < budget:
                    reservoir[j] = item
            seen_nonjudged += 1

    all_docs = kept + reservoir

    # ---- mapping id texte BEIR -> entier + corpus DataFrame ------------
    id_map: dict[str, int] = {}
    records = []
    for int_id, (beir_id, text) in enumerate(all_docs):
        id_map[beir_id] = int_id
        records.append({"doc_id": int_id, "subject": "", "body": text, "text": text})
    corpus = pd.DataFrame(records)

    # ---- requetes + relevants remappes ---------------------------------
    queries = []
    for query_id, rel_docs in qrels.items():
        relevant = sorted(id_map[d] for d in rel_docs if d in id_map)
        if relevant:  # on garde les requetes ayant au moins un pertinent indexe
            queries.append({"query": query_text[query_id], "relevant": relevant})

    return corpus, queries


def main():
    print(f">> Telechargement et conversion de FiQA ({config.FIQA_DOCS_DATASET})...")
    corpus, queries = load_fiqa()

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(config.CORPUS_PARQUET, index=False)
    with open(config.QUERIES_JSON, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    n_rel = sum(len(q["relevant"]) for q in queries)
    print(f"   Corpus : {len(corpus)} documents -> {config.CORPUS_PARQUET}")
    print(f"   Requetes : {len(queries)} ({n_rel} jugements) -> {config.QUERIES_JSON}")
    print("\nEtape suivante : python src/build_index.py --dataset fiqa")


if __name__ == "__main__":
    main()
