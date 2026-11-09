# GDEx

Coptic dictionary example retrieval and learned ranking. 

```sh
pip install -r requirements.txt
pip install -e . --no-deps
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
python scripts/audit_reference_counts.py
python scripts/train_token_senses.py
python scripts/compare_forest.py
python scripts/forest_retrieval_metrics.py
python scripts/crum_length_stats.py
python run.py serve --port 8012
```

Inputs: TT corpora, CCL XML, TAB annotations and Crum transcription. Optional BGE vectors use `requirements-semantic.txt` and `scripts/cache_embeddings.py --sentences`.

Search uses random forest inclusion, competing-entry scores and LightGBM preference order. TAB exclusions and source preferences supply training labels. Source provenance stays visible.

Data, models and reports remain local. Tests: `python -m unittest discover -s tests -q`.

Recommendations use a 30-Coptic-letter cap. Edit [dialect restrictions](dialect_bias.md) and [regression vocabulary](regression_vocabulary.md) on GitHub.
