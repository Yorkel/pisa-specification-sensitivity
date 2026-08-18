"""Model construction, fitting and importance measurement for one grid cell."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import PERMUTATION_REPEATS, RANDOM_SEED

# Hyperparameters are held fixed across the grid. The study varies specification
# choices, not tuning effort; letting tuning vary would confound the axes with
# search budget. Pruning settings follow the shallow configurations that the
# validation partition favours for this data.
TREE_PARAMS = {"max_depth": 5, "min_samples_leaf": 50, "ccp_alpha": 0.001}
FOREST_PARAMS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20, "n_jobs": -1}
BOOST_PARAMS = {"max_iter": 200, "max_depth": 3, "learning_rate": 0.05, "l2_regularization": 1.0}


def build_model(model_family: str, target_form: str, seed: int = RANDOM_SEED):
    """Return an unfitted estimator for the given family and target form."""
    classify = target_form == "binary"
    if model_family == "pruned_tree":
        cls = DecisionTreeClassifier if classify else DecisionTreeRegressor
        return cls(random_state=seed, **TREE_PARAMS)
    if model_family == "random_forest":
        cls = RandomForestClassifier if classify else RandomForestRegressor
        return cls(random_state=seed, **FOREST_PARAMS)
    if model_family == "gradient_boosting":
        cls = HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor
        return cls(random_state=seed, **BOOST_PARAMS)
    raise ValueError(f"Unknown model family: {model_family}")


def fit_model(model, x, y, weights, weighting: str):
    """Fit, applying survey weights only when the specification asks for them.

    Under 'unweighted' the sample weights are withheld entirely rather than set
    to a constant, which is the choice a researcher ignoring the survey design
    would make.
    """
    if weighting == "weighted":
        model.fit(x, y, sample_weight=weights)
    else:
        model.fit(x, y)
    return model


def scoring_name(target_form: str) -> str:
    """The scorer permutation importance is measured against."""
    return "roc_auc" if target_form == "binary" else "r2"


def feature_importances(
    model,
    x_eval,
    y_eval,
    weights,
    target_form: str,
    feature_names: list[str],
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Permutation importance measured on the held-out partition.

    Impurity-based importance is computed differently by trees, forests and
    boosters, so it cannot be compared across the model family axis. Permutation
    importance asks the same question of every model: how much does performance
    fall when this feature is shuffled. It is measured on held-out data because
    importance on the training partition rewards memorisation.
    """
    result = permutation_importance(
        model,
        x_eval,
        y_eval,
        scoring=scoring_name(target_form),
        n_repeats=PERMUTATION_REPEATS,
        random_state=seed,
        n_jobs=-1,
        sample_weight=weights,
    )
    importances = np.asarray(result.importances_mean, dtype=float)
    if importances.shape[0] != len(feature_names):
        raise ValueError("Importance vector does not match the feature count")
    return importances


def grouped_permutation_importance(
    model,
    x_eval,
    y_eval,
    weights,
    target_form: str,
    blocks: dict[str, list[str]],
    metric_fn,
    n_repeats: int = PERMUTATION_REPEATS,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Permutation importance for blocks of features shuffled together.

    Permuting a feature on its own is uninformative when a near-duplicate remains
    in the matrix: the model reads the substitute and barely degrades. ESCS is
    constructed by OECD from HISEI, HOMEPOS and PAREDINT, so those four cannot be
    separated by single-feature permutation at all.

    Every column in a block receives the *same* row permutation. That preserves
    the correlation structure within the block while breaking the block's
    relationship to the outcome, which is the quantity of interest.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    if target_form == "binary":
        baseline = metric_fn(y_eval, model.predict_proba(x_eval)[:, 1], weights)
    else:
        baseline = metric_fn(y_eval, model.predict(x_eval), weights)

    result: dict[str, float] = {}
    for name, columns in blocks.items():
        present = [c for c in columns if c in x_eval.columns]
        if not present:
            continue
        drops = []
        for _ in range(n_repeats):
            shuffled = x_eval.copy()
            order = rng.permutation(len(shuffled))
            shuffled[present] = shuffled[present].to_numpy()[order]
            if target_form == "binary":
                score = metric_fn(y_eval, model.predict_proba(shuffled)[:, 1], weights)
            else:
                score = metric_fn(y_eval, model.predict(shuffled), weights)
            drops.append(baseline - score)
        result[name] = float(np.mean(drops))
    return result
