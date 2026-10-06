import numpy as np
from scipy import stats
from scipy.sparse.csgraph import connected_components
from statsmodels.stats import contingency_tables
from statsmodels.stats.multitest import multipletests

PERMUTATIONS = 10000
NO_CHANGE = {"statistic": 0.0, "df": None, "p_value": 1.0, "note": "all paired differences are zero"}


def mcnemar(ref, cf) -> dict:
    table = [[int(((ref == a) & (cf == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    if not table[0][1] + table[1][0]:
        return {"test": "mcnemar"} | NO_CHANGE
    r = contingency_tables.mcnemar(table, exact=False, correction=False)
    return {"test": "mcnemar", "statistic": float(r.statistic), "df": 1, "p_value": float(r.pvalue)}


def wilcoxon(d) -> dict:
    base = {"test": "wilcoxon_signed_rank"}
    if not np.any(d):
        return base | NO_CHANGE
    r = stats.wilcoxon(d)
    return base | {"statistic": float(r.statistic), "df": None, "p_value": float(r.pvalue)}


def paired_t(d) -> dict:
    base = {"test": "paired_t_test"}
    if not np.any(d):
        return base | NO_CHANGE
    r = stats.ttest_1samp(d, 0.0)
    return base | {"statistic": float(r.statistic), "df": len(d) - 1, "p_value": float(r.pvalue)}


def stuart_maxwell(ref, cf, categories: list, seed: int) -> dict:
    base = {"test": "stuart_maxwell"}
    changed = ref != cf
    cats = [c for c in categories if c in set(ref[changed]) | set(cf[changed])]
    if not cats:
        return base | NO_CHANGE
    code, k = {c: i for i, c in enumerate(cats)}, len(cats)
    r, c = (np.array([code[v] for v in x[changed]]) for x in (ref, cf))
    table = lambda a, b: np.bincount(a * k + b, minlength=k * k).reshape(k, k)

    n, label = connected_components(table(r, c), directed=False)
    groups = [np.flatnonzero(label == i) for i in range(n)]
    statistic = lambda a, b: sum(contingency_tables.SquareTable(table(a, b)[np.ix_(g, g)], shift_zeros=False)
                                 .homogeneity().statistic for g in groups)
    p = stats.permutation_test((r, c), statistic, permutation_type="samples", alternative="greater",
                               n_resamples=PERMUTATIONS, rng=seed).pvalue if len(r) > 1 else 1.0
    return base | {"statistic": float(statistic(r, c)), "df": k - n, "p_value": float(p)}


def adjust(p):
    return multipletests(p, method="fdr_bh")[1]
