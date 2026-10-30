# GDEx: Coptic Dictionary Example Recommendation

GDEx is a research prototype for retrieving, ranking, and selecting corpus examples for Coptic lexicography.

Given a TLA lexical entry and dialect, the system identifies candidate examples and ranks them according to linguistic and lexicographic criteria. A local web interface displays Coptic sentences, English translations, source references, and ranking evidence.

## Features

- TLA entry and lemma lookup
- Sahidic and Bohairic support
- Feature-based example ranking
- Selection of syntactically diverse example sets
- Source attribution and ranking details
- Ablation and evaluation experiments

## Method

The Sahidic research ranker uses 45 features:

| Feature group | Features |
| --- | ---: |
| GDEX surface features | 24 |
| Dependency-based features | 12 |
| Information-related features | 8 |
| Gloss overlap | 1 |

Training uses two complementary sources of supervision:

**TAB:** Manually selected examples provide evidence of editorial inclusion. Unselected sentences are not automatically treated as negative examples.

**Crum:** Historical dictionary example ordering provides a weak signal of editorial preference rather than a direct measure of example quality.

LightGBM models learn rankings from these features and supervision signals. The main inference pipeline does not require a large language model or BGE embeddings.

In addition to ranking individual examples, GDEx compares selecting the top three examples with selecting three examples for greater syntactic coverage.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/wjhuah/GDEx.git
cd GDEx

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Data preparation

**The repository contains source code, not the underlying research data, generated indexes, or trained model weights.**

To reproduce the system, users must obtain and prepare the necessary resources separately, subject to the relevant licenses and access permissions.

Required sources include:

1. [Coptic SCRIPTORIUM corpora](https://github.com/CopticScriptorium/Corpora)
2. Comprehensive Coptic Lexicon XML
3. Manually curated TAB dictionary examples
4. Crum dictionary material and references

The original development environment uses a source directory containing:

```text
coptic_dictionary/
├── corpora/
├── Comprehensive_Coptic_Lexicon-v1.2-2020.xml
├── citations_manual_merged.tab
└── Crum_plain_text/
```

Researchers should use their own authorized copies of these resources. TAB annotations and processed Crum material may need to be obtained from the project collaborators.

Configure the source-data location:

```bash
export COPTIC_SOURCE_ROOT=/path/to/coptic_dictionary
```

Build the corpus indexes and models:

```bash
python run.py build
python run.py dataset
python scripts/audit_crum_bible.py
python run.py train
```

Some experimental comparisons require additional historical resources and locally generated reports. Consult the technical documentation for details.

## Web interface

After preparing the data and compatible models:

```bash
python run.py serve --port 8012
```

Open http://localhost:8012.

Search for a TLA ID (such as `C998`), choose the dialect and ranking method, and inspect the recommended examples.

The Evaluation page also requires generated experiment reports that are not included in this repository.

## Evaluation

The Sahidic full-feature model achieved approximately **0.785 NDCG@6** on the evaluated TAB inclusion-ranking task.

This measures ranking among labeled candidate groups, not full-corpus retrieval accuracy.

Bohairic results remain provisional due to limited annotated supervision.

## Repository structure

- `src/`: retrieval, feature extraction, ranking, and web interface
- `scripts/`: data preparation, training, and evaluation
- `cluster/`: example-set selection experiments
- `assets/`: web interface styles
- `docs/`: research and technical documentation
- `tests/`: automated tests
- `reports/`: selected research notes

## Limitations

GDEx is intended to assist expert lexicographic review, not replace it.

Wordform matching does not guarantee correct sense identification, particularly for homographs. Automatic syntactic coverage is an approximation of example diversity.

The current results require further independent evaluation and expert review.

## Resources

- [Coptic Dictionary Online](https://coptic-dictionary.org/)
- [Coptic SCRIPTORIUM](https://copticscriptorium.org/)
- [Coptic SCRIPTORIUM Corpora](https://github.com/CopticScriptorium/Corpora)

## Status

Research prototype. Full corpus data and trained models are not distributed with the source code.
