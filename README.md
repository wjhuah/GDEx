# GDEx

Coptic dictionary example retrieval and ranking, with separate dialects and token-level highlighting.

## Run

With the local corpus index and trained models already prepared:

```sh
python -m pip install -e .
export COPTIC_SOURCE_ROOT=/path/to/coptic_dictionary
python run.py serve --port 8012
```

Open <http://127.0.0.1:8012>. Search by Coptic spelling or TLA ID.

Corpora, indexes and trained models are prepared locally. Generated reports and logs stay local.

Tests: `python -m unittest discover -s tests -q`.
