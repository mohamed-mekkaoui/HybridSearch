"""Construction et chargement des index BM25 et semantique (FAISS).

Les artefacts sont mis en cache sur disque (artifacts/) afin de ne les
calculer qu'une seule fois. La recherche devient alors quasi instantanee.
"""
from __future__ import annotations

import pickle
import re
from typing import List

import numpy as np

import config

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# Caches paresseux pour les ressources optionnelles (stop-words, normaliseur).
_stopwords_cache = None
_normalizer_cache = None


def _get_stopwords():
    """Ensemble des mots vides anglais (sklearn, aucun telechargement)."""
    global _stopwords_cache
    if _stopwords_cache is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _stopwords_cache = set(ENGLISH_STOP_WORDS)
    return _stopwords_cache


def _get_word_normalizer():
    """Renvoie une fonction mot -> mot selon config.BM25_WORD_NORMALIZATION.

    "stemming" : PorterStemmer (nltk). "lemmatization" : WordNetLemmatizer
    (nltk + donnees WordNet, telechargees automatiquement au besoin).
    """
    global _normalizer_cache
    if _normalizer_cache is not None:
        return _normalizer_cache

    mode = config.BM25_WORD_NORMALIZATION
    try:
        if mode == "stemming":
            from nltk.stem import PorterStemmer

            stemmer = PorterStemmer()
            _normalizer_cache = stemmer.stem
        elif mode == "lemmatization":
            import nltk
            from nltk.stem import WordNetLemmatizer

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
            _normalizer_cache = lemmatizer.lemmatize
        else:
            raise ValueError(
                f"BM25_WORD_NORMALIZATION inconnu : {mode!r} "
                "(attendu: none / stemming / lemmatization)."
            )
    except ImportError as exc:
        raise ImportError(
            f"BM25_WORD_NORMALIZATION={mode!r} necessite nltk. "
            "Installez-le (`pip install nltk`) ou mettez la valeur a 'none'."
        ) from exc

    return _normalizer_cache


def tokenize(text: str) -> List[str]:
    """Tokenisation BM25 configurable via config.

    Etapes (selon la config) : decoupage alphanumerique -> minuscules ->
    filtre des mots vides -> stemming OU lemmatisation -> filtre longueur min.
    NB: partagee entre l'index et la requete ; changer un reglage impose de
    reconstruire l'index BM25.
    """
    tokens = _TOKEN_RE.findall(text.lower() if config.BM25_LOWERCASE else text)

    if config.BM25_REMOVE_STOPWORDS:
        stop = _get_stopwords()
        tokens = [t for t in tokens if t.lower() not in stop]

    if config.BM25_WORD_NORMALIZATION != "none":
        normalize = _get_word_normalizer()
        tokens = [normalize(t) for t in tokens]

    if config.BM25_MIN_TOKEN_LEN > 1:
        tokens = [t for t in tokens if len(t) >= config.BM25_MIN_TOKEN_LEN]

    return tokens


# --------------------------------------------------------------------------
# Index BM25
# --------------------------------------------------------------------------
def build_bm25(corpus_texts: List[str]):
    """Construit un index BM25Okapi a partir des textes du corpus."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized, k1=config.BM25_K1, b=config.BM25_B)


def save_bm25(bm25) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PICKLE, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25():
    with open(config.BM25_PICKLE, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# Embeddings + index FAISS
# --------------------------------------------------------------------------
_model_cache = {}


def get_model(model_name: str = config.EMBEDDING_MODEL):
    """Charge (et met en cache) le modele SentenceTransformer."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: List[str], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    """Encode une liste de textes en vecteurs L2-normalises (float32).

    La normalisation permet d'utiliser le produit scalaire (IndexFlatIP)
    comme similarite cosinus exacte.
    """
    model = get_model(model_name)
    emb = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def embed_query(query: str, model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    """Encode une requete unique en vecteur L2-normalise (shape: [dim])."""
    model = get_model(model_name)
    vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    return vec[0].astype("float32")


def build_faiss(embeddings: np.ndarray):
    """Construit un index FAISS de produit scalaire (= cosinus si normalise)."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_embeddings(embeddings: np.ndarray) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_NPY, embeddings)


def save_faiss(index) -> None:
    import faiss

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX))


def load_embeddings() -> np.ndarray:
    return np.load(config.EMBEDDINGS_NPY)


def load_faiss():
    import faiss

    return faiss.read_index(str(config.FAISS_INDEX))


def artifacts_exist() -> bool:
    return (
        config.BM25_PICKLE.exists()
        and config.EMBEDDINGS_NPY.exists()
        and config.FAISS_INDEX.exists()
    )
