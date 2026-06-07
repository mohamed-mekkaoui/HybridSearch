"""Script : construit le corpus puis les index BM25 et semantique (FAISS).

Usage :
    python src/build_index.py [--dataset {enron,fiqa}] [--force]

--dataset selectionne le dataset (defaut: enron).
--force reconstruit tout meme si le cache existe.

Pour FiQA, le corpus est genere au prealable par `python src/load_ir_dataset.py`.
"""
from __future__ import annotations

import argparse
import time

import config
import indexing
import pandas as pd


def get_corpus(force: bool) -> "pd.DataFrame":
    """Charge le corpus selon le dataset actif (config.DATASET)."""
    if config.DATASET == "enron":
        import preprocessing

        return preprocessing.load_or_build_corpus(force=force)

    # Autres datasets (ex: fiqa) : le corpus est produit par un loader dedie.
    if not config.CORPUS_PARQUET.exists():
        raise FileNotFoundError(
            f"Corpus introuvable pour le dataset '{config.DATASET}' : "
            f"{config.CORPUS_PARQUET}\n"
            "Lance d'abord le loader, par ex. `python src/load_ir_dataset.py`."
        )
    return pd.read_parquet(config.CORPUS_PARQUET)


def main(force: bool = False):
    t0 = time.perf_counter()
    print(f">> Dataset actif : {config.DATASET}")

    print(">> Etape 1/3 : chargement du corpus...")
    corpus = get_corpus(force=force)
    print(f"   Corpus : {len(corpus)} documents")

    print(">> Etape 2/3 : index BM25...")
    if config.BM25_PICKLE.exists() and not force:
        print("   (cache existant, ignore)")
    else:
        bm25 = indexing.build_bm25(corpus["text"].tolist())
        indexing.save_bm25(bm25)
        print("   BM25 sauvegarde.")

    print(">> Etape 3/3 : embeddings + index FAISS...")
    if config.FAISS_INDEX.exists() and not force:
        print("   (cache existant, ignore)")
    else:
        embeddings = indexing.embed_texts(corpus["text"].tolist())
        indexing.save_embeddings(embeddings)
        faiss_index = indexing.build_faiss(embeddings)
        indexing.save_faiss(faiss_index)
        print(f"   Embeddings {embeddings.shape} + FAISS sauvegardes.")

    print(f"\nTermine en {time.perf_counter() - t0:.1f}s. Artefacts dans {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=config.SUPPORTED_DATASETS, default=config.DATASET,
        help="dataset a indexer (defaut: enron)",
    )
    parser.add_argument("--force", action="store_true", help="reconstruire le cache")
    args = parser.parse_args()
    config.configure(args.dataset)
    main(force=args.force)
