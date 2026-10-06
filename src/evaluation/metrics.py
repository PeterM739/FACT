import warnings

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, equal_opportunity_difference, equalized_odds_difference, selection_rate
from scipy import stats

LEVEL, RESAMPLES = 0.95, 2000


def wilson(k: int, n: int) -> tuple[float, float]:
    ci = stats.binomtest(k, n).proportion_ci(LEVEL, method="wilson")
    return float(ci.low), float(ci.high)


def bootstrap(data: tuple, stat, seed: int) -> tuple[float, float]:
    if len(data[0]) < 2:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ci = stats.bootstrap(data, stat, paired=True, method="percentile", confidence_level=LEVEL, n_resamples=RESAMPLES,
                             rng=seed).confidence_interval
    return float(ci.low), float(ci.high)


def rank_biserial(d) -> float:
    d = d[d != 0]
    if not len(d):
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def rate_ratio(ref, cf, axis=-1):
    return cf.mean(axis) / ref.mean(axis)


def category_shift(ref, cf, k: int):
    """Change in the share of each of the k categories (counterfactual − reference); ref and cf are category codes."""
    return (np.bincount(cf, minlength=k) - np.bincount(ref, minlength=k)) / len(ref)


def total_variation(ref, cf, k: int) -> float:
    return 0.5 * np.abs(category_shift(ref, cf, k)).sum()


def group_fairness(ref, cf) -> dict:
    """The two conditions are the groups, the reference answer is the label."""
    data = {"y_true": np.tile(ref, 2).astype(int), "y_pred": np.concatenate([ref, cf]).astype(int),
            "sensitive_features": np.repeat(["reference", "counterfactual"], len(ref))}
    rate = MetricFrame(metrics=selection_rate, **data).by_group
    return {"effect_value": rate["counterfactual"] - rate["reference"],
            "rate_reference": rate["reference"], "rate_counterfactual": rate["counterfactual"],
            "rate_ratio": rate["counterfactual"] / rate["reference"] if rate["reference"] else np.nan,
            "equal_opportunity_difference": equal_opportunity_difference(**data) if ref.any() else np.nan,
            "equalized_odds_difference": equalized_odds_difference(**data) if ref.any() and not ref.all() else np.nan}


def transitions(ref, cf, categories: list) -> pd.DataFrame:
    table = pd.crosstab(pd.Categorical(ref, categories), pd.Categorical(cf, categories), dropna=False)
    counts = table.stack().rename("count").reset_index()
    counts.columns = ["original", "counterfactual", "count"]
    counts["row_pct"] = (table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100).stack().to_numpy()
    counts["col_pct"] = (table.div(table.sum(axis=0).replace(0, np.nan), axis=1) * 100).stack().to_numpy()
    return counts
