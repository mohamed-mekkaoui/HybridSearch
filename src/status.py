"""Affiche l'etat des index de chaque dataset (utile pendant les builds).

Un dataset est TERMINE & COHERENT quand les 4 artefacts existent ET que le
nombre d'embeddings correspond au nombre de documents du corpus (sinon les
fichiers viennent de builds differents).

Usage :
    python src/status.py
"""
from __future__ import annotations

import os

import numpy as np
import pyarrow.parquet as pq

import config


def main():
    for dataset in config.SUPPORTED_DATASETS:
        config.configure(dataset)
        corpus, emb, faiss = config.CORPUS_PARQUET, config.EMBEDDINGS_NPY, config.FAISS_INDEX

        if not corpus.exists():
            print(f"[{dataset}] aucun corpus (rien construit)")
            continue

        n_corpus = pq.read_table(str(corpus)).num_rows
        n_emb = np.load(emb, mmap_mode="r").shape[0] if emb.exists() else 0
        has_faiss = faiss.exists()
        coherent = (n_emb == n_corpus) and has_faiss

        etat = "TERMINE & COHERENT" if coherent else "EN COURS / INCOMPLET"
        print(f"[{dataset}] corpus={n_corpus} embeddings={n_emb} faiss={has_faiss} -> {etat}")


if __name__ == "__main__":
    main()
