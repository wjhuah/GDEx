# GDEx

Corpus example retrieval and ranking for the Coptic Dictionary Online.

Search by TLA ID or Coptic lemma, choose a dialect, and compare individual examples and sets of three. Ambiguous lemmas open a list of entries with their meanings and forms. Target words are highlighted using TT token boundaries.

## Models

The system learns two related tasks:

- **Inclusion:** TAB selections and Crum citations are positive examples. TAB exclusions provide negative examples. Crum citation position does not change the inclusion label.
- **Preference:** TAB priorities and the order of citations within Crum sections provide ranking supervision.

Both models use LightGBM. Features cover sentence form, dependency structure, target context, definition similarity, competing senses, and corpus collocations. Lexical overlap is one input; semantic similarity also supports paraphrases in English translations. The optional cross-sense experiment constructs negative entry–sentence pairs for homographs.

Search results retain recorded TAB and Crum selections first. Crum order is preserved within each source section. Remaining candidates are ranked by the learned models. Sets of three balance recommendation score, dependency coverage and repetition.

Sahidic and Bohairic have separate candidate pools, frequency tables and models. Dialects are registered in `dialects.toml`. Attested additional forms and their dialects are recorded in `lexical_forms.json`.

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

Use the project copies of the lexicon, TAB annotations and Crum transcription. TT corpora are available from [Coptic SCRIPTORIUM](https://github.com/CopticScriptorium/Corpora). Data, caches, trained models and experiment reports stay outside version control.

```sh
export COPTIC_SOURCE_ROOT=/path/to/coptic_dictionary
python run.py build
python scripts/audit_crum_bible.py
python run.py dataset
python run.py train
python scripts/build_typicality.py
python scripts/prepare_features.py
python scripts/train_inclusion.py
python run.py serve --port 8012
```

Open http://localhost:8012. To reuse the project's BGE vectors, place `embeddings.sqlite3` in `cache/` before extracting features. Missing vectors are recorded as missing values. Computing new embeddings requires the optional packages in `requirements-semantic.txt` and a locally cached `BAAI/bge-large-en-v1.5` model.

## Development

```sh
python -m unittest discover -s tests -q
```

- `src/coptic_examples/corpus.py`: TT indexing
- `src/coptic_examples/crum_audit.py`: Crum reference extraction and alignment
- `src/coptic_examples/sense.py`: token matching and POS alignment
- `src/coptic_examples/inclusion.py`: entry-conditioned features and prediction
- `src/coptic_examples/typicality.py`: dependency frame and collocation counts
- `src/coptic_examples/editorial.py`: recorded source selections and partial orders
- `scripts/train_inclusion.py`: training and grouped evaluation
- `src/coptic_examples/ui.py`: search and comparison pages

Evaluation uses spelling-family folds and removes repeated held-out Coptic spans from training. Experimental results are written locally under `reports/`. The web interface also exposes local evaluation results when available.
