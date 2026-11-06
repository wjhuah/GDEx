# GDEx

Example retrieval and ranking for the Coptic Dictionary Online.

Search by TLA ID or Coptic lemma and choose a dialect. Ambiguous lemmas open a list of entries with their definitions and attested forms. Highlighting follows native TT token boundaries. Damaged original spans retain their source markers; marked short fragments paired with long translations are excluded from model examples and recommendations. Dictionary subcategories are retained, with title senses distinguished in English definitions.

## Learning tasks

The interface compares two LightGBM designs:

- **Inclusion + ranking:** a binary inclusion classifier and a separate continuous preference regressor.
- **Unified model:** a single continuous model learns negative examples and graded source preferences. Unranked positive selections have a lower-bound loss.

TAB selections and all uniquely aligned Crum citations are positive inclusion examples. TAB exclusions, including `forbidden_urns=ALL`, have label −1. Binary training maps −1 to class 0. TAB priorities use 3, 2, 1; Crum preferences decrease evenly from 3 to 1.5 across aligned examples of an entry. Inclusion labels stay positive throughout that range.

Augmentation experiments compare shared-token homograph negatives, unrelated examples and mixtures at several sizes. Known positive pairs are protected. Evaluation holds out spelling families and removes repeated held-out Coptic texts from training; generated examples come from the training fold only. Model selection uses these development folds.

Features include sentence form, native dependencies, definition overlap, cached BGE cosine, competing senses and corpus collocations. Recorded source selections appear first, with available TAB/Crum partial orders preserved; learned scores order new candidates. Evaluation tables report model predictions before this source-order restoration. Augmentation comparisons use a fixed threshold; a separate development grid chooses the deployed inclusion operating point.

Sahidic and Bohairic have separate pools, frequencies and models. `dialects.toml` registers dialects. `lexical_forms.json`, `lexical_conditions.json` and `token_alignments.json` record additional forms and reviewed alignments while retaining original TT annotations.

## Installation

Requires Python 3.11 or later.

```sh
git clone https://github.com/wjhuah/GDEx.git
cd GDEx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Data and training

Set `COPTIC_SOURCE_ROOT` to a directory containing:

```text
corpora/
Comprehensive_Coptic_Lexicon-v1.2-2020.xml
citations_manual_merged.tab
Crum_plain_text/
```

Use the project copies of the lexicon, TAB annotations and Crum transcription. TT corpora are available from [Coptic SCRIPTORIUM](https://github.com/CopticScriptorium/Corpora). The current [CDO entry index](https://coptic-dictionary.org/index.json) supplies definitions and dialect forms by TLA ID. Its snapshot is stored locally with a source hash. Updating it requires retraining.

```sh
export COPTIC_SOURCE_ROOT=/path/to/coptic_dictionary
python scripts/sync_dictionary.py
python run.py build
python scripts/audit_crum_bible.py
python run.py dataset
python run.py train
python scripts/build_typicality.py
python scripts/prepare_features.py
python scripts/compare_learning.py
python scripts/select_thresholds.py
python scripts/audit_learning_senses.py
python scripts/ablate_learning.py
python scripts/check_negative_weights.py
python run.py serve --port 8012
```

Open http://localhost:8012. `/experiments` displays the current model comparison and all feature importances. `/sense-review` shows the five spelling groups and held-out source diagnostics. Reports, data, caches and model files stay outside version control.

For optional semantic features, install `requirements-semantic.txt` and cache `BAAI/bge-large-en-v1.5` locally. Before model comparison, reuse `cache/embeddings.sqlite3` or run:

```sh
python scripts/cache_embeddings.py --sentences
```

Inference reads cached vectors. Missing vectors are explicit missing values. Embedding coverage affects results; local reports record the evaluated data and settings.

## Development

```sh
python -m unittest discover -s tests -q
```

- `corpus.py`: TT indexing and native annotations
- `lexicon.py`: TLA-specific definitions and dialect forms
- `crum_audit.py`: complete explicit Bible-marker ledger and alignment
- `sense.py`: target-token matching and reviewed alignments
- `inclusion.py`: entry-conditioned feature extraction
- `learning.py`: source labels, augmentation and both model designs
- `editorial.py`: recorded source selections and partial orders
- `scripts/compare_learning.py`: fixed-fold comparisons and final training

Training exports are written to `data/derived/model_rows_<dialect>.jsonl`, with the ordered input columns in `feature_schema.json` and validation assignments in `fold_manifest_<dialect>.json`. `check_negative_weights.py` compares class balancing with class-and-entry balancing on the same folds.
