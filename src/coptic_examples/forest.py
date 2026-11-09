"""Random forest inclusion with training-only missing-value imputation."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .learning import sample_weights


def fit(rows):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=4,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(
        np.asarray([r["x"] for r in rows], dtype=float),
        [r["y"] > 0 for r in rows],
        forest__sample_weight=sample_weights(rows),
    )
    return model
