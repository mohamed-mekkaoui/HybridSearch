"""Outil d'annotation manuelle de la pertinence (ground-truth qrels).

Pour chaque requete, on construit un "pool" de candidats (union des top-K des
trois methodes BM25 / semantique / hybride), puis l'utilisateur coche les
emails reellement pertinents. Les jugements sont ecrits dans data/queries.json
(champ "relevant"), que run_eval.py utilisera ensuite en priorite.

Lancement :
    streamlit run src/annotate.py

IMPORTANT : les doc_id dependent du corpus courant (corpus.parquet). Annotez
APRES avoir construit l'index sur le dataset final (ex: le vrai Enron). Si vous
reconstruisez le corpus avec d'autres parametres, les doc_id changent et les
annotations doivent etre refaites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import config
import indexing
from search import SearchEngine

st.set_page_config(page_title="Annotation de pertinence", layout="wide")

# Profondeur du pool par methode (nombre de candidats a juger par requete)
POOL_K = 20


@st.cache_resource(show_spinner="Chargement des index...")
def get_engine(dataset: str) -> SearchEngine:
    config.configure(dataset)
    return SearchEngine.from_artifacts()


def load_queries() -> list[dict]:
    with open(config.QUERIES_JSON, encoding="utf-8") as f:
        return json.load(f)


def save_queries(queries: list[dict]) -> None:
    with open(config.QUERIES_JSON, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


@st.cache_data(show_spinner=False)
def build_pool(dataset: str, query: str, pool_k: int) -> list[int]:
    """Union ordonnee des top-K des 3 methodes pour la requete."""
    engine = get_engine(dataset)
    seen: dict[int, None] = {}
    for method in ("bm25", "semantic", "hybrid"):
        for r in engine.search(query, method=method, top_k=pool_k).results:
            seen.setdefault(r.doc_id, None)
    return list(seen.keys())


# --------------------------------------------------------------------------
st.title("📝 Annotation manuelle de la pertinence")

dataset = st.sidebar.selectbox(
    "Dataset", config.SUPPORTED_DATASETS,
    index=config.SUPPORTED_DATASETS.index(config.DATASET),
)
config.configure(dataset)

if not indexing.artifacts_exist():
    st.error(f"Index introuvables pour '{dataset}'. Lancez d'abord `python src/build_index.py --dataset {dataset}`.")
    st.stop()

engine = get_engine(dataset)
queries = load_queries()

# Etat persistant des jugements : {query: set(doc_id)}
if "judgments" not in st.session_state:
    st.session_state.judgments = {
        item["query"]: set(item.get("relevant", [])) for item in queries
    }

# --- Choix de la requete --------------------------------------------------
query_list = [item["query"] for item in queries]
with st.sidebar:
    st.header("Requetes")
    selected = st.radio("Requete a annoter", query_list, index=0)
    st.divider()
    st.caption("Avancement")
    for q in query_list:
        n = len(st.session_state.judgments.get(q, set()))
        st.write(f"{'✅' if n else '⬜'} {q} — {n} pertinents")
    st.divider()
    if st.button("💾 Sauvegarder dans queries.json", type="primary"):
        for item in queries:
            item["relevant"] = sorted(st.session_state.judgments.get(item["query"], set()))
        save_queries(queries)
        st.success("Annotations sauvegardees.")

# --- Liste des candidats a juger ------------------------------------------
st.subheader(f"Requete : « {selected} »")
st.caption(
    f"Cochez les emails PERTINENTS. Pool = union des top-{POOL_K} des 3 methodes. "
    "Pensez a cliquer « Sauvegarder » dans la barre laterale."
)

pool = build_pool(dataset, selected, POOL_K)
current = st.session_state.judgments.setdefault(selected, set())

st.write(f"**{len(pool)} candidats** — **{len(current)} marques pertinents**")

for doc_id in pool:
    subject = engine.corpus.at[doc_id, "subject"] or "(sans sujet)"
    body = engine.corpus.at[doc_id, "body"]
    col_chk, col_txt = st.columns([1, 12])
    with col_chk:
        checked = st.checkbox(
            "Pertinent", value=(doc_id in current), key=f"{selected}_{doc_id}",
            label_visibility="collapsed",
        )
    with col_txt:
        with st.expander(f"#{doc_id} — {subject}"):
            st.write(body[:1500] + ("..." if len(body) > 1500 else ""))
    if checked:
        current.add(doc_id)
    else:
        current.discard(doc_id)
