"""Moteur de recherche : BM25, semantique et hybride.

Le SearchEngine charge les index (BM25 + FAISS) et le corpus, puis expose
une methode `search(query, method, top_k)` renvoyant les resultats classes
ainsi que le temps de recherche.

Methodes hybrides disponibles :
  - 'weighted' : alpha * BM25_norm + (1-alpha) * sem_norm  (normalisation config: minmax/zscore)
  - 'rrf'      : Reciprocal Rank Fusion ponderee (alpha sur BM25, 1-alpha sur le
                 semantique ; alpha=0.5 = RRF standard a poids egaux)

Les defauts (mode de fusion, normalisation, alpha) proviennent de config.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

import config
import indexing


@dataclass
class SearchResult:
    doc_id: int
    score: float
    subject: str
    body: str


@dataclass
class SearchResponse:
    results: List[SearchResult]
    elapsed_ms: float
    method: str


def _normalize(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalise un vecteur de scores selon la methode choisie.

    - "minmax" : ramene dans [0, 1].
    - "zscore" : centre-reduit (moyenne 0, ecart-type 1).
    Robuste au cas degenere (scores constants).
    """
    if method == "minmax":
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    if method == "zscore":
        mean, std = scores.mean(), scores.std()
        if std < 1e-12:
            return np.zeros_like(scores)
        return (scores - mean) / std

    raise ValueError(f"Methode de normalisation inconnue : {method!r}")


class SearchEngine:
    def __init__(
        self,
        corpus: pd.DataFrame,
        bm25,
        embeddings: np.ndarray,
        faiss_index,
        model_name: str = config.EMBEDDING_MODEL,
    ):
        self.corpus = corpus.reset_index(drop=True)
        self.bm25 = bm25
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.model_name = model_name
        self.n_docs = len(self.corpus)

    # ---- Chargement pratique depuis le cache disque -----------------------
    @classmethod
    def from_artifacts(cls) -> "SearchEngine":
        import preprocessing

        corpus = preprocessing.load_or_build_corpus()
        bm25 = indexing.load_bm25()
        embeddings = indexing.load_embeddings()
        faiss_index = indexing.load_faiss()
        return cls(corpus, bm25, embeddings, faiss_index)

    # ---- Scores bruts par methode ----------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        """Score BM25 pour TOUS les documents du corpus."""
        return np.asarray(self.bm25.get_scores(indexing.tokenize(query)), dtype="float32")

    def semantic_scores(self, query: str) -> np.ndarray:
        """Similarite cosinus requete/document pour TOUS les documents."""
        q = indexing.embed_query(query, self.model_name)
        # embeddings deja normalises => produit scalaire = cosinus
        return (self.embeddings @ q).astype("float32")

    # ---- Recherche unifiee ------------------------------------------------
    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 10,
        alpha: Optional[float] = None,
        hybrid_mode: Optional[str] = None,
        normalization: Optional[str] = None,
    ) -> SearchResponse:
        alpha = config.DEFAULT_ALPHA if alpha is None else alpha
        hybrid_mode = hybrid_mode or config.DEFAULT_HYBRID_MODE
        normalization = normalization or config.NORMALIZATION
        start = time.perf_counter()

        if method == "bm25":
            scores = self.bm25_scores(query)
            order = np.argsort(-scores)[:top_k]
            ranked = [(int(i), float(scores[i])) for i in order]

        elif method == "semantic":
            scores = self.semantic_scores(query)
            order = np.argsort(-scores)[:top_k]
            ranked = [(int(i), float(scores[i])) for i in order]

        elif method == "hybrid":
            ranked = self._hybrid(query, top_k, alpha, hybrid_mode, normalization)

        else:
            raise ValueError(f"Methode inconnue : {method!r}")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results = [
            SearchResult(
                doc_id=doc_id,
                score=score,
                subject=self.corpus.at[doc_id, "subject"],
                body=self.corpus.at[doc_id, "body"],
            )
            for doc_id, score in ranked
        ]
        return SearchResponse(results=results, elapsed_ms=elapsed_ms, method=method)

    def _hybrid(self, query: str, top_k: int, alpha: float, mode: str, normalization: str):
        bm25 = self.bm25_scores(query)
        sem = self.semantic_scores(query)

        if mode == "weighted":
            fused = alpha * _normalize(bm25, normalization) + (1 - alpha) * _normalize(sem, normalization)
            order = np.argsort(-fused)[:top_k]
            return [(int(i), float(fused[i])) for i in order]

        elif mode == "rrf":
            depth = config.FUSION_DEPTH
            bm25_rank = {int(d): r for r, d in enumerate(np.argsort(-bm25)[:depth])}
            sem_rank = {int(d): r for r, d in enumerate(np.argsort(-sem)[:depth])}
            k = config.RRF_K
            # RRF pondere : alpha sur BM25, (1-alpha) sur le semantique.
            # alpha=0.5 reproduit le RRF standard (poids egaux) au facteur d'echelle pres.
            w_bm25, w_sem = alpha, 1.0 - alpha
            candidates = set(bm25_rank) | set(sem_rank)
            fused = {}
            for d in candidates:
                s = 0.0
                if d in bm25_rank:
                    s += w_bm25 * 1.0 / (k + bm25_rank[d] + 1)
                if d in sem_rank:
                    s += w_sem * 1.0 / (k + sem_rank[d] + 1)
                fused[d] = s
            ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
            return [(int(d), float(s)) for d, s in ranked]

        raise ValueError(f"Mode hybride inconnu : {mode!r}")
