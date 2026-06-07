"""Configuration centrale du projet.

Tous les chemins et hyperparametres sont definis ici pour faciliter
la reproductibilite et eviter les valeurs magiques dispersees dans le code.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Chemins de base (independants du dataset)
# --------------------------------------------------------------------------
# Racine du projet (le dossier qui contient `src/`)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_BASE = ROOT_DIR / "artifacts"   # racine des caches (sous-dossier par dataset)
RESULTS_BASE = ROOT_DIR / "results"       # racine des resultats (sous-dossier par dataset)

# Dataset Enron brut (CSV Kaggle : colonnes `file`, `message`)
RAW_ENRON_CSV = DATA_DIR / "emails.csv"

# Datasets supportes
SUPPORTED_DATASETS = ("enron", "fiqa")

# --------------------------------------------------------------------------
# Selection du dataset actif + chemins associes
# --------------------------------------------------------------------------
# Les artefacts de chaque dataset vivent dans un sous-dossier dedie afin de ne
# pas se melanger (artifacts/enron/, artifacts/fiqa/, results/<dataset>/...).
# Tous les modules lisent ces variables AU MOMENT DE L'APPEL (pas a l'import),
# donc appeler configure(<dataset>) en debut de script redirige tout le pipeline.

# Variables peuplees par configure() ci-dessous (declarees pour la lisibilite).
DATASET = "enron"
ARTIFACTS_DIR = ARTIFACTS_BASE / DATASET
RESULTS_DIR = RESULTS_BASE / DATASET
CORPUS_PARQUET = ARTIFACTS_DIR / "corpus.parquet"
BM25_PICKLE = ARTIFACTS_DIR / "bm25.pkl"
EMBEDDINGS_NPY = ARTIFACTS_DIR / "embeddings.npy"
FAISS_INDEX = ARTIFACTS_DIR / "faiss.index"
QUERIES_JSON = DATA_DIR / "queries.json"
QRELS_JSON = ARTIFACTS_DIR / "qrels.json"


def configure(dataset: str) -> None:
    """Selectionne le dataset actif et recalcule tous les chemins associes."""
    global DATASET, ARTIFACTS_DIR, RESULTS_DIR, CORPUS_PARQUET
    global BM25_PICKLE, EMBEDDINGS_NPY, FAISS_INDEX, QUERIES_JSON, QRELS_JSON

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset inconnu : {dataset!r}. Choix : {', '.join(SUPPORTED_DATASETS)}."
        )

    DATASET = dataset
    ARTIFACTS_DIR = ARTIFACTS_BASE / dataset
    RESULTS_DIR = RESULTS_BASE / dataset
    CORPUS_PARQUET = ARTIFACTS_DIR / "corpus.parquet"
    BM25_PICKLE = ARTIFACTS_DIR / "bm25.pkl"
    EMBEDDINGS_NPY = ARTIFACTS_DIR / "embeddings.npy"
    FAISS_INDEX = ARTIFACTS_DIR / "faiss.index"
    QRELS_JSON = ARTIFACTS_DIR / "qrels.json"
    # Requetes : Enron utilise le jeu manuel ; FiQA son fichier genere.
    QUERIES_JSON = DATA_DIR / ("queries.json" if dataset == "enron" else f"queries_{dataset}.json")


# Dataset actif par defaut (surchargeable par la variable d'env PROJECT_DATASET).
configure(os.environ.get("PROJECT_DATASET", "enron"))

# --------------------------------------------------------------------------
# Pretraitement
# --------------------------------------------------------------------------
# Nombre max d'emails a conserver dans le corpus (echantillonnage).
# Enron contient ~500k mails ; on en garde quelques milliers comme demande.
MAX_EMAILS = 5000
# Longueur minimale du corps (en caracteres) pour garder un email
MIN_BODY_CHARS = 30
# Longueur max conservee par email (troncature pour l'embedding)
MAX_BODY_CHARS = 4000
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Indexation BM25 : tokenisation + parametres
# --------------------------------------------------------------------------
# La tokenisation est partagee entre l'index (corpus) et la requete. Apres
# avoir change un de ces flags, RECONSTRUIRE l'index BM25 (cf. README).
BM25_LOWERCASE = True            # passer le texte en minuscules
BM25_REMOVE_STOPWORDS = False    # retirer les mots vides anglais (sklearn)
BM25_MIN_TOKEN_LEN = 1           # longueur minimale d'un token conserve
# Normalisation des mots (necessite nltk si != "none") :
#   "none"          : aucune (defaut)
#   "stemming"      : racinisation Porter, rapide, agressive (energy -> energi)
#   "lemmatization" : lemmatisation WordNet, vrais mots (studies -> study)
BM25_WORD_NORMALIZATION = "none"
# Parametres de la formule BM25Okapi (valeurs standard par defaut).
BM25_K1 = 1.5                    # saturation de la frequence des termes
BM25_B = 0.75                    # normalisation par la longueur du document

# --------------------------------------------------------------------------
# Dataset FiQA (benchmark BEIR annote, charge via ir_datasets)
# --------------------------------------------------------------------------
# Identifiants ir_datasets : corpus dans le split racine, requetes+qrels dans /test.
FIQA_DOCS_DATASET = "beir/fiqa"
FIQA_QRELS_DATASET = "beir/fiqa/test"
# Plafond du nombre de documents indexes (les docs juges dans les qrels sont
# TOUJOURS conserves en plus, pour ne pas fausser le Recall).
FIQA_MAX_DOCS = 20000

# --------------------------------------------------------------------------
# Modele d'embeddings semantiques
# --------------------------------------------------------------------------
# Alternatives possibles (changer le modele => reconstruire embeddings + FAISS) :
#   "multi-qa-MiniLM-L6-cos-v1"   : optimise recherche (asymetrique), 384 dim (defaut)
#   "all-MiniLM-L6-v2"            : generaliste rapide, 384 dim
#   "multi-qa-mpnet-base-cos-v1"  : plus precis mais plus lent, 768 dim
#   "all-mpnet-base-v2"           : generaliste de qualite, 768 dim
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_BATCH_SIZE = 64

# --------------------------------------------------------------------------
# Recherche hybride
# --------------------------------------------------------------------------
# Mode de fusion par defaut : "weighted" (ponderee) ou "rrf".
DEFAULT_HYBRID_MODE = "weighted"
# Methode de normalisation des scores avant fusion ponderee : "minmax" ou "zscore".
NORMALIZATION = "minmax"
# Poids de la fusion : alpha sur BM25, (1-alpha) sur le semantique.
# Utilise par le mode "weighted" ET par le mode "rrf" (RRF pondere ;
# alpha=0.5 => RRF standard a poids egaux).
DEFAULT_ALPHA = 0.5
# Constante de la Reciprocal Rank Fusion
RRF_K = 60
# Profondeur de fusion (on fusionne les top-N de chaque methode)
FUSION_DEPTH = 100

# Ces reglages (mode, normalisation, alpha) s'appliquent A LA RECHERCHE :
# aucune reconstruction d'index necessaire, il suffit de relancer run_eval.py.

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
EVAL_K_VALUES = (5, 10)
# Profondeur de pooling pour generer les pseudo-qrels (top-K de chaque methode)
POOL_DEPTH = 10


def snapshot() -> dict:
    """Capture les reglages d'experimentation courants (pour l'historique des runs)."""
    return {
        "dataset": DATASET,
        "embedding_model": EMBEDDING_MODEL,
        "bm25_lowercase": BM25_LOWERCASE,
        "bm25_remove_stopwords": BM25_REMOVE_STOPWORDS,
        "bm25_word_normalization": BM25_WORD_NORMALIZATION,
        "bm25_min_token_len": BM25_MIN_TOKEN_LEN,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "hybrid_mode": DEFAULT_HYBRID_MODE,
        "normalization": NORMALIZATION,
        "alpha": DEFAULT_ALPHA,
        "rrf_k": RRF_K,
        "fusion_depth": FUSION_DEPTH,
    }


def auto_label() -> str:
    """Petit identifiant lisible decrivant la config (si aucun label fourni)."""
    return (
        f"wn={BM25_WORD_NORMALIZATION},stop={int(BM25_REMOVE_STOPWORDS)},"
        f"k1={BM25_K1},b={BM25_B},mode={DEFAULT_HYBRID_MODE},"
        f"norm={NORMALIZATION},alpha={DEFAULT_ALPHA}"
    )
