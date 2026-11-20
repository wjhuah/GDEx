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

## Evaluation

Five outer spelling-family folds; two inner folds; eight parameter settings per algorithm and stage. Duplicate Coptic texts are excluded across training and validation. Threshold: 0.5. Metrics use the original unfiltered labeled candidate sets; retrieval uses a maximum of 20 whitespace-separated groups.

| Classification model | Features | AUC | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 62 | 0.9337 | 0.7770 |
| Random Forest, nested-tuned | 15 | 0.9170 | 0.7651 |

| Ranking model | Features | TAB NDCG@6 | Crum NDCG@6 |
|---|---:|---:|---:|
| LightGBM | 62 | 0.9510 | 0.9013 |
| LightGBM, nested-tuned | 15 | 0.9570 | 0.8980 |

The search did not improve every metric. Existing deployed weights are retained.


Feature definitions: [FEATURES.md](FEATURES.md).

Annotation packets are generated locally with `python scripts/export_annotation_html.py /path/to/local/selection`. Open either HTML file, rank examples from 1 to 5, and export answers as JSON.

Reproduce tuning after preparing the local labeled datasets:

```sh
python scripts/prepare_compact_matrix.py cache/tuning_matrix.joblib
python scripts/nested_compact.py cache/tuning_matrix.joblib
```
