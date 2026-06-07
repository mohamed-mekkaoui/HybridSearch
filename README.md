# Moteur de recherche — BM25 vs Sémantique vs Hybride

Mini-étude expérimentale de **Recherche d'Information** comparant trois approches de
recherche, mesurées avec les métriques standard du domaine (Precision@K, Recall@K, MRR,
NDCG@K) sur deux jeux de données.

1. **BM25** — recherche lexicale par mots-clés (`rank-bm25`)
2. **Sémantique** — embeddings Sentence-Transformers + FAISS (similarité cosinus)
3. **Hybride** — fusion des deux, par pondération normalisée **ou** Reciprocal Rank Fusion (RRF)

## Fonctionnalités

- **3 méthodes de recherche** comparables sur le même corpus.
- **2 datasets** : Enron (emails réels, via kagglehub) et FiQA (benchmark BEIR annoté, via `ir_datasets`).
- **Hybride configurable** : fusion pondérée (normalisation min-max ou z-score) **et** RRF pondéré.
- **Prétraitement réglable** : minuscules, stop-words, stemming / lemmatisation, paramètres BM25 `k1`/`b`.
- **Évaluation rigoureuse** : P@K, Recall@K, MRR, NDCG@K, temps de réponse.
- **Historique des runs** + script de comparaison pour mesurer l'effet de chaque réglage.
- **Interface Streamlit** : page de recherche + page de benchmark.
- **Index mis en cache** sur disque : construits une fois, recherche quasi instantanée.

## Démarrage rapide (FiQA, évaluation rigoureuse)

```powershell
pip install -r requirements.txt
python src/load_ir_dataset.py                # télécharge FiQA (corpus + requêtes + qrels)
python src/build_index.py --dataset fiqa     # construit BM25 + embeddings + FAISS
python src/run_eval.py --dataset fiqa --label "baseline"
streamlit run src/app.py                     # interface (recherche + benchmark)
```

> Les dossiers `data/`, `artifacts/` et `results/` sont **générés localement** et exclus du
> dépôt (cf. `.gitignore`). Après un clone, lancez les scripts ci-dessus pour les recréer.

## Exemple de résultats (FiQA, 648 requêtes annotées)

| Méthode | Precision@10 | Recall@10 | MRR | NDCG@10 | Temps moyen |
|---|---|---|---|---|---|
| BM25 | 0.083 | 0.367 | 0.364 | 0.298 | ~85 ms |
| Sémantique | 0.131 | 0.543 | 0.541 | 0.465 | ~25 ms |
| Hybride | 0.119 | 0.506 | 0.500 | 0.426 | ~95 ms |

Sur FiQA (questions souvent reformulées), la recherche **sémantique dépasse BM25** : peu de
mots exacts en commun, l'avantage va au sens. Les chiffres dépendent des réglages et se
régénèrent avec `run_eval.py`.

## Structure

```
projet_ihm/
├── data/                      # (généré) datasets + requêtes/qrels
├── src/
│   ├── config.py             # configuration centrale + sélection du dataset actif
│   ├── preprocessing.py      # parsing + nettoyage des emails Enron -> corpus
│   ├── download_data.py      # téléchargement Enron via kagglehub
│   ├── load_ir_dataset.py    # chargeur FiQA (BEIR) annoté via ir_datasets
│   ├── indexing.py           # tokenisation + index BM25 + embeddings + FAISS
│   ├── search.py             # SearchEngine : bm25 / semantic / hybrid
│   ├── evaluation.py         # P@K, Recall@K, MRR, NDCG@K, temps + pseudo-qrels
│   ├── visualization.py      # graphiques comparatifs + projection t-SNE/UMAP
│   ├── build_index.py        # script : construit corpus + index (--dataset)
│   ├── run_eval.py           # script : évalue + historise + figures (--dataset, --label)
│   ├── compare_runs.py       # script : compare les runs de l'historique
│   ├── status.py             # script : état des index par dataset
│   ├── annotate.py           # outil d'annotation manuelle (Streamlit)
│   └── app.py                # interface : recherche + benchmark (Streamlit)
├── artifacts/<dataset>/       # (généré) caches : corpus, BM25, embeddings, FAISS
├── results/<dataset>/         # (généré) <label>/ par run (métriques+figures) + history.jsonl
├── requirements.txt
└── README.md
```

## Les deux datasets

Chaque dataset a ses propres artefacts (`artifacts/<dataset>/`) et résultats
(`results/<dataset>/`), sélectionnés par `--dataset` :

- **`enron`** (défaut) — vrais emails, **sans annotations** → démo de recherche + évaluation
  indicative par pseudo-qrels (ou annotation manuelle via `annotate.py`).
- **`fiqa`** — benchmark financier **livré avec ses qrels officiels** → évaluation rigoureuse.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation détaillée

### Dataset Enron (téléchargement automatique via kagglehub)

Le pipeline télécharge le dataset au premier `build_index.py` si `data/emails.csv` est absent.
Il faut `kagglehub` installé et une authentification Kaggle :

- soit le fichier `~/.kaggle/kaggle.json` (token API depuis votre compte Kaggle),
- soit la variable d'environnement `KAGGLE_API_TOKEN` (ou `KAGGLE_USERNAME` / `KAGGLE_KEY`).

Le dataset (~1,4 Go) est mis en cache dans `~/.cache/kagglehub/` (pas recopié dans `data/`).

```powershell
python src/build_index.py --dataset enron    # corpus + index BM25 + FAISS
python src/run_eval.py --dataset enron        # évaluation + figures
streamlit run src/app.py                      # interface
```

### Dataset FiQA (annotations officielles)

```powershell
pip install ir_datasets
python src/load_ir_dataset.py                # FiQA -> corpus + queries_fiqa.json
python src/build_index.py --dataset fiqa
python src/run_eval.py --dataset fiqa
```

> FiQA est plafonné à `FIQA_MAX_DOCS = 20000` docs dans `config.py` (les documents jugés dans
> les qrels sont toujours conservés). L'embedding prend quelques minutes sur CPU.

## Choix techniques et améliorations

| Sujet | Décision |
|---|---|
| Fusion hybride | BM25 et cosinus n'ont pas la même échelle : on **normalise les scores (min-max ou z-score)** avant la pondération. Alternative : **RRF** (sur les rangs), pondérable par `alpha`. |
| Similarité cosinus | Embeddings **L2-normalisés** + `faiss.IndexFlatIP` ⇒ produit scalaire = cosinus exact. |
| Prétraitement | Tokenisation configurable : stop-words, **stemming** (Porter) ou **lemmatisation** (WordNet). |
| Performance | Embeddings et index **mis en cache** : construits une fois, recherche quasi instantanée. |
| Jugements de pertinence | FiQA : qrels officiels. Enron : pooling semi-automatique ou annotation manuelle. |
| Modèle | `multi-qa-MiniLM-L6-cos-v1`, optimisé pour la recherche question/document. |
| Reproductibilité | seed fixe, échantillonnage déterministe, versions épinglées. |

## Expérimentation : régler les algorithmes

Tous les réglages sont centralisés dans [src/config.py](src/config.py). Selon ce que vous
modifiez, il faut (ou non) reconstruire les index.

| Ce que vous changez | Où | Réindexation |
|---|---|---|
| `alpha`, `RRF_K`, `FUSION_DEPTH`, `DEFAULT_HYBRID_MODE`, `NORMALIZATION` | `config.py` | Non — relancer `run_eval.py` |
| `BM25_K1`, `BM25_B` | `config.py` | BM25 seulement |
| Tokenisation (`BM25_REMOVE_STOPWORDS`, `BM25_WORD_NORMALIZATION`, …) | `config.py` | BM25 seulement |
| `EMBEDDING_MODEL` | `config.py` | Embeddings + FAISS seulement |
| Nettoyage du texte / taille du corpus (`MAX_EMAILS`, …) | `preprocessing.py`, `config.py` | Tout |

### Réindexation partielle (éviter de tout recalculer)

L'embedding est lent ; BM25 est quasi instantané. On reconstruit seulement ce qui change :

```powershell
# Changement BM25 seul (tokenisation, k1/b) : reconstruit BM25, garde les embeddings
Remove-Item artifacts\<dataset>\bm25.pkl
python src/build_index.py --dataset <dataset>        # SANS --force

# Changement du modèle d'embedding : ré-embed, garde BM25
Remove-Item artifacts\<dataset>\embeddings.npy, artifacts\<dataset>\faiss.index
python src/build_index.py --dataset <dataset>        # SANS --force

# Changement du prétraitement / du corpus : tout reconstruire
python src/build_index.py --dataset <dataset> --force
```

> `BM25_WORD_NORMALIZATION` accepte `"none"`, `"stemming"` ou `"lemmatization"` (ces deux
> dernières nécessitent `nltk`). Après tout changement de tokenisation, **reconstruire BM25**.

### Comparer les expériences (historique des runs)

Chaque `run_eval.py` sauvegarde ses métriques et figures dans un **dossier dédié**
`results/<dataset>/<label>/` (les images de chaque expérience sont donc conservées), et
ajoute une ligne au journal global `results/<dataset>/history.jsonl`. On étiquette le run
avec `--label`, puis on compare :

```powershell
python src/run_eval.py --dataset fiqa --label "baseline"
# ... après un changement de config ...
python src/run_eval.py --dataset fiqa --label "avec stemming"

python src/compare_runs.py --dataset fiqa            # tableaux par métrique + history_comparison.md
```

## Annotation manuelle (Enron)

Enron n'a pas d'annotations. Un outil permet d'en créer :

```powershell
python src/build_index.py --dataset enron
streamlit run src/annotate.py     # cocher les emails pertinents par requête
python src/run_eval.py --dataset enron
```

Les `doc_id` sont écrits dans `data/queries.json`. Dès qu'au moins une requête est annotée,
`run_eval.py` utilise ces jugements au lieu des pseudo-qrels.

> Les `doc_id` dépendent du corpus courant : annotez **après** avoir figé l'index.

## Limites

L'évaluation par pooling lexical (Enron, sans annotations) favorise mécaniquement BM25. Les
chiffres rigoureux de comparaison proviennent donc de **FiQA** (qrels officiels) ; Enron sert
d'illustration concrète sur des emails réels.

## Technologies

Python · `rank-bm25` · `sentence-transformers` · `faiss-cpu` · `ir_datasets` (BEIR) ·
`pandas` · `numpy` · `scikit-learn` · `nltk` · `matplotlib` / `seaborn` · `streamlit`.
