"""Interface Streamlit : recherche + benchmark comparatif.

Lancement :
    streamlit run src/app.py

Deux pages (barre laterale) :
  - Recherche : saisir une requete, choisir la methode, voir le Top-K.
  - Benchmark : regler les parametres, lancer l'evaluation des 3 methodes,
    afficher les graphiques de comparaison (metriques + temps).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permet `streamlit run src/app.py` de trouver les modules du dossier src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

import config
import evaluation
import indexing
import visualization
from search import SearchEngine

st.set_page_config(page_title="Moteur de recherche", layout="wide")

METHOD_LABEL = {"bm25": "BM25", "semantic": "Semantique", "hybrid": "Hybride"}
EXAMPLES = {
    "enron": ["budget report", "energy trading", "meeting schedule", "contract negotiation"],
    "fiqa": ["how to invest in stocks", "tax on capital gains", "credit card debt", "mortgage refinancing"],
}


@st.cache_resource(show_spinner="Chargement des index et du modele...")
def get_engine(dataset: str) -> SearchEngine:
    config.configure(dataset)
    return SearchEngine.from_artifacts()


# ==========================================================================
# Page 1 : Recherche
# ==========================================================================
def page_search(engine: SearchEngine, dataset: str):
    st.header("Recherche")

    with st.sidebar:
        st.subheader("Parametres de recherche")
        method_label = st.radio("Methode", ["BM25", "Semantique", "Hybride"], index=2)
        method = {"BM25": "bm25", "Semantique": "semantic", "Hybride": "hybrid"}[method_label]
        top_k = st.slider("Top-K", 1, 20, 10)

        alpha = config.DEFAULT_ALPHA
        hybrid_mode = config.DEFAULT_HYBRID_MODE
        normalization = config.NORMALIZATION
        if method == "hybrid":
            hybrid_mode = st.selectbox(
                "Mode de fusion", ["weighted", "rrf"],
                format_func=lambda m: "Ponderee (alpha)" if m == "weighted" else "Reciprocal Rank Fusion",
            )
            alpha = st.slider(
                "alpha (poids BM25 vs semantique)", 0.0, 1.0, config.DEFAULT_ALPHA, 0.05,
                help="0 = tout semantique, 1 = tout BM25. Pour RRF, 0.5 = poids egaux.",
            )
            if hybrid_mode == "weighted":
                normalization = st.selectbox(
                    "Normalisation", ["minmax", "zscore"],
                    index=["minmax", "zscore"].index(config.NORMALIZATION),
                )

    query = st.text_input("Votre requete", placeholder="ex: energy trading...")
    cols = st.columns(4)
    for col, ex in zip(cols, EXAMPLES.get(dataset, [])):
        if col.button(ex):
            query = ex

    if not query:
        st.info("Saisissez une requete ou cliquez sur un exemple ci-dessus.")
        return

    resp = engine.search(
        query, method=method, top_k=top_k, alpha=alpha,
        hybrid_mode=hybrid_mode, normalization=normalization,
    )
    st.success(f"{len(resp.results)} resultats — temps : **{resp.elapsed_ms:.1f} ms**")
    for rank, r in enumerate(resp.results, start=1):
        with st.expander(f"#{rank} — {r.subject or '(sans sujet)'}  ·  score={r.score:.4f}"):
            st.write(r.body[:1500] + ("..." if len(r.body) > 1500 else ""))
            st.caption(f"doc_id = {r.doc_id}")


# ==========================================================================
# Page 2 : Benchmark
# ==========================================================================
def _summary_table(summary: dict) -> pd.DataFrame:
    metrics = ["P@5", "P@10", "Recall@5", "Recall@10", "MRR", "NDCG@10", "time_mean_ms"]
    rows = {}
    for method, row in summary.items():
        rows[METHOD_LABEL.get(method, method)] = {m: round(row.get(m, 0.0), 4) for m in metrics}
    df = pd.DataFrame(rows).T
    return df.rename(columns={"time_mean_ms": "Temps moyen (ms)"})


def page_benchmark(engine: SearchEngine, dataset: str):
    st.header("Benchmark — comparaison des 3 strategies")
    st.caption(
        "Lance l'evaluation (Precision, Recall, MRR, NDCG, temps) sur les requetes "
        "annotees du dataset, puis compare BM25, Semantique et Hybride."
    )

    queries_raw = evaluation.load_queries()
    n_queries = len(queries_raw)

    with st.sidebar:
        st.subheader("Parametres du benchmark")
        max_queries = st.slider(
            "Nombre de requetes evaluees", 1, n_queries, min(100, n_queries),
            help="Sous-echantillonner accelere le benchmark (surtout sur FiQA).",
        )
        st.markdown("**Reglages de l'hybride**")
        hybrid_mode = st.selectbox(
            "Mode de fusion", ["weighted", "rrf"],
            format_func=lambda m: "Ponderee (alpha)" if m == "weighted" else "Reciprocal Rank Fusion",
        )
        normalization = config.NORMALIZATION
        alpha = st.slider(
            "alpha (poids BM25 vs semantique)", 0.0, 1.0, config.DEFAULT_ALPHA, 0.05,
            help="0 = tout semantique, 1 = tout BM25. Pour RRF, 0.5 = poids egaux.",
        )
        if hybrid_mode == "weighted":
            normalization = st.selectbox("Normalisation", ["minmax", "zscore"])
        run = st.button("Lancer le benchmark", type="primary")

    if run:
        with st.spinner("Preparation des jugements de pertinence..."):
            queries_with_rels, manual = evaluation.prepare_qrels(engine, queries_raw)
        queries_with_rels = queries_with_rels[:max_queries]

        with st.spinner(f"Evaluation des 3 methodes sur {len(queries_with_rels)} requetes..."):
            summary = evaluation.evaluate(
                engine, queries_with_rels,
                alpha=alpha, hybrid_mode=hybrid_mode, normalization=normalization,
            )
        # Persistance pour rester affiche apres rerun
        st.session_state["bench"] = {
            "summary": summary, "dataset": dataset, "manual": manual,
            "n": len(queries_with_rels),
            "params": {"mode": hybrid_mode, "alpha": alpha, "norm": normalization},
        }

    bench = st.session_state.get("bench")
    if not bench or bench["dataset"] != dataset:
        st.info("Regle les parametres puis clique sur **Lancer le benchmark**.")
        return

    summary = bench["summary"]
    src = "annotations officielles" if bench["manual"] else "pseudo-qrels (pooling)"
    p = bench["params"]
    st.write(
        f"**{bench['n']} requetes** — jugements : {src} — "
        f"hybride : {p['mode']}"
        + (f", alpha={p['alpha']}, norm={p['norm']}" if p["mode"] == "weighted" else "")
    )

    # Tableau recapitulatif
    st.subheader("Resultats")
    st.dataframe(_summary_table(summary), use_container_width=True)

    # Graphiques (reutilise visualization.py -> PNG dans results/<dataset>/)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metriques")
        path = visualization.plot_metric_comparison(summary)
        st.image(str(path), use_column_width=True)
    with col2:
        st.subheader("Temps de recherche")
        path = visualization.plot_time_comparison(summary)
        st.image(str(path), use_column_width=True)

    # Meilleure methode par metrique (lecture rapide)
    st.subheader("Meilleure methode par metrique")
    best = {}
    for metric in ["P@10", "Recall@10", "MRR", "NDCG@10"]:
        winner = max(summary, key=lambda m: summary[m].get(metric, 0.0))
        best[metric] = f"{METHOD_LABEL[winner]} ({summary[winner].get(metric, 0.0):.3f})"
    st.table(pd.DataFrame([best]).T.rename(columns={0: "Gagnant"}))


# ==========================================================================
# Routage
# ==========================================================================
st.title("Moteur de recherche — BM25 vs Semantique vs Hybride")

page = st.sidebar.radio("Page", ["Recherche", "Benchmark"])
dataset = st.sidebar.selectbox(
    "Dataset", config.SUPPORTED_DATASETS,
    index=config.SUPPORTED_DATASETS.index(config.DATASET),
)
config.configure(dataset)

if not indexing.artifacts_exist():
    st.error(
        f"Index introuvables pour '{dataset}'.\n\n"
        f"Construis-les d'abord : `python src/build_index.py --dataset {dataset}` "
        "(pour fiqa : `python src/load_ir_dataset.py` au prealable)."
    )
    st.stop()

engine = get_engine(dataset)
st.caption(f"Dataset : {dataset} — {engine.n_docs} documents — modele : {engine.model_name}")

if page == "Recherche":
    page_search(engine, dataset)
else:
    page_benchmark(engine, dataset)
