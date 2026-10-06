import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import (bootstrap, category_shift, group_fairness, rank_biserial, rate_ratio,
                                    total_variation, wilson)
from src.evaluation.statistical_tests import adjust, mcnemar, paired_t, stuart_maxwell, wilcoxon
from src.generation.design import comparisons

ALPHA = 0.05
NO_CI = (np.nan, np.nan)


def _ci(prefix: str, bounds: tuple) -> dict:
    return {f"{prefix}_ci_lower": bounds[0], f"{prefix}_ci_upper": bounds[1]}


def _shifts(d) -> dict:
    return {"flips": int(np.count_nonzero(d)), "n_lower": int((d < 0).sum()), "n_higher": int((d > 0).sum())}


def analyze_binary(ref, cf, var, seed) -> dict:
    d = cf - ref
    n, k, n1, n10 = len(d), int(np.count_nonzero(d)), int(ref.sum()), int((d < 0).sum())
    boot = lambda data, f: bootstrap(data, f, seed)
    fair = group_fairness(ref, cf)
    return {"metric": "counterfactual_flip_rate", "metric_value": k / n, **_ci("metric", wilson(k, n)),
            "flips": k, "n01": int((d > 0).sum()), "n10": n10,
            "effect_size": "paired_proportion_difference", "effect_value": fair.pop("effect_value"),
            **_ci("effect", boot((d,), np.mean)), **fair,
            **_ci("rate_reference", wilson(n1, n)), **_ci("rate_counterfactual", wilson(int(cf.sum()), n)),
            **_ci("rate_ratio", boot((ref, cf), rate_ratio) if n1 else NO_CI),
            **_ci("equal_opportunity_difference", wilson(n10, n1) if n1 else NO_CI),
            **mcnemar(ref, cf)}


def analyze_ordinal(ref, cf, var, seed) -> dict:
    d = cf - ref
    n, s = len(d), _shifts(d)
    boot = lambda f: bootstrap((d,), f, seed)
    return {"metric": "change_rate", "metric_value": s["flips"] / n, **_ci("metric", wilson(s["flips"], n)),
            "rate_lower": s["n_lower"] / n, **_ci("rate_lower", wilson(s["n_lower"], n)),
            "rate_higher": s["n_higher"] / n, **_ci("rate_higher", wilson(s["n_higher"], n)),
            "effect_size": "mean_counterfactual_difference", "effect_value": d.mean(), **_ci("effect", boot(np.mean)),
            "rank_biserial": rank_biserial(d), **s,
            **wilcoxon(d)}


def analyze_categorical(ref, cf, var, seed) -> dict:
    n, k, m = len(ref), int((ref != cf).sum()), len(var["values"])
    code = {c: i for i, c in enumerate(var["values"])}
    r, c = (np.array([code[v] for v in x]) for x in (ref, cf))
    shift = category_shift(r, c, m)
    top = int(np.abs(shift).argmax())
    boot = lambda f: bootstrap((r, c), f, seed)
    return {"metric": "change_rate", "metric_value": k / n, **_ci("metric", wilson(k, n)), "flips": k,
            "effect_size": "total_variation_distance", "effect_value": total_variation(r, c, m),
            **_ci("effect", boot(lambda a, b: total_variation(a, b, m))),
            "largest_shift_category": var["values"][top] if shift.any() else None,
            "largest_shift": shift[top] if shift.any() else None,
            **_ci("largest_shift", boot(lambda a, b: category_shift(a, b, m)[top]) if shift.any() else NO_CI),
            **stuart_maxwell(ref, cf, var["values"], seed)}


def analyze_continuous(ref, cf, var, seed) -> dict:
    d = cf - ref
    boot = lambda x: bootstrap((x,), np.mean, seed)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return {"metric": "mean_absolute_difference", "metric_value": np.abs(d).mean(), **_ci("metric", boot(np.abs(d))),
            "effect_size": "mean_difference", "effect_value": d.mean(), **_ci("effect", boot(d)),
            "d_z": d.mean() / sd if sd else np.nan, **_shifts(d),
            **paired_t(d)}


ANALYZERS = {"binary": analyze_binary, "ordinal": analyze_ordinal, "categorical": analyze_categorical,
             "continuous": analyze_continuous}
RESULT_COLUMNS = ["metric", "metric_value", "metric_ci_lower", "metric_ci_upper", "effect_size", "effect_value",
                  "effect_ci_lower", "effect_ci_upper", "test", "statistic", "df", "p_value", "note"]


def _design_groups(pairs: pd.DataFrame, cfg):
    for model in (m.name for m in cfg.models):
        for comp in comparisons(cfg):
            mask = (pairs["model"] == model) & (pairs["sensitive_attribute"] == comp.sensitive_attribute) & (pairs["comparison"] == comp.label)
            yield (model, comp.sensitive_attribute, comp.label), pairs[mask]


def run_tests(pairs: pd.DataFrame, variables: list[dict], cfg, n_cases: int) -> pd.DataFrame:
    rows, groups = [], list(_design_groups(pairs, cfg))
    bar = tqdm(total=len(groups) * len(variables), desc="analyze    tests", unit="test", leave=False)
    for (model, sensitive_attribute, comparison), g in groups:
        for var in variables:
            both = g[[f"{var['name']}__ref", f"{var['name']}__cf"]].dropna()
            row = {"model": model, "sensitive_attribute": sensitive_attribute, "comparison": comparison, "variable": var["name"],
                   "variable_type": var["type"],
                   "n_pairs": len(both), "n_pairs_invalid": n_cases - len(g), "n_pairs_undetermined": len(g) - len(both)}
            if len(both):
                ref, cf = both.to_numpy().T
                if var["type"] != "categorical":
                    ref, cf = ref.astype(float), cf.astype(float)
                row |= ANALYZERS[var["type"]](ref, cf, var, cfg.generation.seed)
            else:
                row["note"] = "no valid pairs"
            rows.append(row)
            bar.update()
    bar.close()
    tests = pd.DataFrame(rows)
    tests = tests.reindex(columns=list(dict.fromkeys([*tests.columns, *RESULT_COLUMNS])))
    tests["family"] = np.where(tests["sensitive_attribute"] == "control", "control", "primary")
    tests["adjusted_p_value"] = np.nan
    for _, idx in tests[tests["p_value"].notna()].groupby("family").groups.items():
        tests.loc[idx, "adjusted_p_value"] = adjust(tests.loc[idx, "p_value"].to_numpy())
    tests["statistically_significant"] = tests["adjusted_p_value"] < ALPHA
    return tests
