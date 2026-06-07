"""Module d'evaluation : metriques de RI + generation des jugements (qrels).

Metriques implementees (jugements binaires) :
  - Precision@K, Recall@K
  - Mean Reciprocal Rank (MRR)
  - NDCG@K
  - Statistiques de temps (moyen / min / max)

Generation des qrels :
  Les jugements de pertinence peuvent etre fournis manuellement dans
  queries.json. A defaut, on utilise une methode SEMI-AUTOMATIQUE par
  "pooling" : on collecte l'union des top-POOL_DEPTH resultats des trois
  methodes pour chaque requete, et on marque pertinents les documents dont
  le texte contient tous les mots de la requete (heuristique lexicale).
  Cette approche reduit le biais d'une seule methode mais reste imparfaite :
  c'est explicitement une evaluation indicative (cf. rapport).
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Sequence

import numpy as np

import config
import indexing
from search import SearchEngine

METHODS = ["bm25", "semantic", "hybrid"]


# --------------------------------------------------------------------------
# Metriques
# --------------------------------------------------------------------------
def precision_at_k(ranked_ids: Sequence[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / k


def recall_at_k(ranked_ids: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[int], relevant: set) -> float:
    for rank, d in enumerate(ranked_ids, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[int], relevant: set, k: int) -> float:
    """NDCG@k avec gains binaires (1 si pertinent, 0 sinon)."""
    top = ranked_ids[:k]
    dcg = sum(
        (1.0 / np.log2(rank + 1)) for rank, d in enumerate(top, start=1) if d in relevant
    )
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, n_rel + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# Generation des qrels (pseudo-pertinence par pooling lexical)
# --------------------------------------------------------------------------
def build_pseudo_qrels(
    engine: SearchEngine,
    queries: List[str],
    pool_depth: int = config.POOL_DEPTH,
) -> Dict[str, List[int]]:
    """Genere des jugements semi-automatiques par pooling des 3 methodes."""
    qrels: Dict[str, List[int]] = {}
    for q in queries:
        pool: set = set()
        for method in METHODS:
            resp = engine.search(q, method=method, top_k=pool_depth)
            pool.update(r.doc_id for r in resp.results)

        q_tokens = set(indexing.tokenize(q))
        relevant = []
        for doc_id in pool:
            doc_tokens = set(indexing.tokenize(engine.corpus.at[doc_id, "text"]))
            # Pertinent si tous les mots de la requete apparaissent dans l'email.
            if q_tokens and q_tokens.issubset(doc_tokens):
                relevant.append(doc_id)
        qrels[q] = sorted(relevant)
    return qrels


def load_queries() -> List[Dict]:
    """Charge queries.json. Format attendu :
        [{"query": "...", "relevant": [optionnel, liste de doc_id]}, ...]
    """
    with open(config.QUERIES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_qrels(engine: SearchEngine, queries_raw: List[Dict]) -> tuple[List[Dict], bool]:
    """Renvoie (queries_with_rels, manuel?).

    Si au moins une requete a des jugements manuels, on les utilise tels quels.
    Sinon on genere des pseudo-qrels par pooling (semi-automatique).
    """
    has_manual = any(item.get("relevant") for item in queries_raw)
    if has_manual:
        return queries_raw, True
    queries = [item["query"] for item in queries_raw]
    qrels = build_pseudo_qrels(engine, queries)
    return [{"query": q, "relevant": qrels[q]} for q in queries], False


# --------------------------------------------------------------------------
# Boucle d'evaluation
# --------------------------------------------------------------------------
def evaluate(
    engine: SearchEngine,
    queries_with_rels: List[Dict],
    k_values: Sequence[int] = config.EVAL_K_VALUES,
    alpha: float = None,
    hybrid_mode: str = None,
    normalization: str = None,
) -> Dict[str, Dict]:
    """Evalue chaque methode sur l'ensemble des requetes.

    Les parametres alpha / hybrid_mode / normalization ne s'appliquent qu'a la
    methode hybride (sinon ignores). None => valeurs par defaut de config.

    Renvoie un dict : {method: {metric_name: valeur_moyenne, ...}}.
    """
    max_k = max(k_values)
    agg: Dict[str, Dict[str, List[float]]] = {
        m: {"times": []} for m in METHODS
    }

    for item in queries_with_rels:
        query = item["query"]
        relevant = set(item.get("relevant", []))
        if not relevant:
            continue  # pas de jugement -> requete ignoree dans les metriques

        for method in METHODS:
            resp = engine.search(
                query, method=method, top_k=max_k,
                alpha=alpha, hybrid_mode=hybrid_mode, normalization=normalization,
            )
            ranked_ids = [r.doc_id for r in resp.results]
            agg[method]["times"].append(resp.elapsed_ms)

            for k in k_values:
                agg[method].setdefault(f"P@{k}", []).append(
                    precision_at_k(ranked_ids, relevant, k)
                )
                agg[method].setdefault(f"Recall@{k}", []).append(
                    recall_at_k(ranked_ids, relevant, k)
                )
                agg[method].setdefault(f"NDCG@{k}", []).append(
                    ndcg_at_k(ranked_ids, relevant, k)
                )
            agg[method].setdefault("MRR", []).append(
                reciprocal_rank(ranked_ids, relevant)
            )

    # Agregation -> moyennes + stats de temps
    summary: Dict[str, Dict] = {}
    for method in METHODS:
        times = agg[method]["times"]
        row = {}
        for key, values in agg[method].items():
            if key == "times":
                continue
            row[key] = float(np.mean(values)) if values else 0.0
        row["time_mean_ms"] = float(np.mean(times)) if times else 0.0
        row["time_min_ms"] = float(np.min(times)) if times else 0.0
        row["time_max_ms"] = float(np.max(times)) if times else 0.0
        summary[method] = row
    return summary
