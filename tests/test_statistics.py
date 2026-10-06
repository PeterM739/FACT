import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from scipy import stats

from src.evaluation.analyze import analyze_binary, analyze_categorical, analyze_ordinal
from src.evaluation.dataset import majority
from src.evaluation.metrics import (bootstrap, category_shift, rank_biserial, rate_ratio, total_variation,
                                    transitions, wilson)
from src.evaluation.statistical_tests import adjust, mcnemar, paired_t, stuart_maxwell, wilcoxon


def test_majority():
    assert majority(pd.Series([1, 1, 0])) == 1
    assert majority(pd.Series(["A"])) == "A"
    assert majority(pd.Series(["A", "B"])) is None


def test_wilson_and_bootstrap():
    assert wilson(3, 10) == pytest.approx((0.10779126740630104, 0.6032218525388546))
    assert wilson(0, 7)[0] == 0.0
    x = np.arange(20.0)
    lo, hi = bootstrap((x,), np.mean, seed=1)
    assert lo < x.mean() < hi and (lo, hi) == bootstrap((x,), np.mean, seed=1)
    assert np.isnan(bootstrap((x[:1],), np.mean, seed=1)).all()


def test_rank_biserial_and_transitions():
    assert rank_biserial(np.array([1.0, 2.0, -1.0, 0.0])) == pytest.approx(0.5)
    tr = transitions(np.array(["A", "A", "B"]), np.array(["A", "B", "B"]), ["A", "B", "C"])
    assert tr["count"].sum() == 3 and len(tr) == 9
    assert tr[tr["original"] == "A"]["row_pct"].sum() == pytest.approx(100)


@pytest.mark.parametrize("n01, n10, statistic", [(3, 9, 3.0), (30, 12, 18 ** 2 / 42), (0, 0, 0.0)])
def test_mcnemar_without_continuity_correction(n01, n10, statistic):
    ref = np.array([0] * n01 + [1] * n10 + [0] * 5 + [1] * 5)
    cf = np.array([1] * n01 + [0] * n10 + [0] * 5 + [1] * 5)
    r = mcnemar(ref, cf)
    assert r["test"] == "mcnemar" and r["statistic"] == pytest.approx(statistic)
    assert r["p_value"] == pytest.approx(stats.chi2.sf(statistic, 1) if statistic else 1.0)


def test_wilcoxon_and_t():
    d = np.array([-2, -1, -1, 0, 1, -3, -2, -1, 0, -1.0])
    assert wilcoxon(d)["p_value"] == pytest.approx(stats.wilcoxon(d).pvalue)
    assert wilcoxon(np.zeros(5))["p_value"] == 1.0
    assert paired_t(d)["p_value"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue)


def test_stuart_maxwell_reference_value():
    rng = np.random.default_rng(0)
    ref = rng.choice(["A", "B", "C"], 200)
    cf = np.where(rng.random(200) < 0.3, rng.choice(["A", "B", "C"], 200, p=[0.6, 0.3, 0.1]), ref)
    r = stuart_maxwell(ref, cf, ["A", "B", "C", "D"], 0)
    assert r["test"] == "stuart_maxwell" and r["df"] == 2 and r["statistic"] == pytest.approx(6.214983713355041)
    assert r["p_value"] == pytest.approx(stats.chi2.sf(r["statistic"], 2), abs=0.01)


def test_stuart_maxwell_with_two_categories_is_mcnemar():
    ref, cf = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 3), np.array(["B"] * 8 + ["A"] * 3 + ["B"] * 9 + ["C"] * 3)
    r, m = stuart_maxwell(ref, cf, ["A", "B", "C"], 0), mcnemar((ref == "A").astype(int), (cf == "A").astype(int))
    assert r["df"] == 1 and r["statistic"] == pytest.approx(m["statistic"])
    assert r["p_value"] == pytest.approx(stats.binomtest(1, 9).pvalue)  # all 2^9 swaps: the exact McNemar test
    assert stuart_maxwell(ref, ref, ["A", "B"], 0)["p_value"] == 1.0


def test_stuart_maxwell_separate_groups_of_categories():
    ref, cf = np.array(["A"] * 4 + ["C"] * 6 + ["B"] * 2), np.array(["B"] * 4 + ["D"] * 6 + ["B"] * 2)
    r = stuart_maxwell(ref, cf, ["A", "B", "C", "D"], 0)
    assert r["test"] == "stuart_maxwell" and r["df"] == 2
    assert r["statistic"] == pytest.approx(10.0) and r["p_value"] == pytest.approx(4 / 2 ** 10)


def test_adjust_is_benjamini_hochberg():
    assert adjust([0.01, 0.04, 0.03, 0.2, 0.001, 0.5]) == pytest.approx([0.03, 0.06, 0.06, 0.24, 0.006, 0.5])


def test_analyzers_report_effects_and_cis():
    ref, cf = np.array([1, 1, 0, 1, 0.0]), np.array([0, 1, 0, 0, 0.0])
    b = analyze_binary(ref, cf, {}, 0)
    assert b["metric_value"] == 0.4 and b["effect_value"] == pytest.approx(-0.4) and b["n10"] == 2
    assert b["metric_ci_lower"] <= 0.4 <= b["metric_ci_upper"] and b["test"] == "mcnemar"
    assert (b["rate_reference"], b["rate_counterfactual"]) == pytest.approx((0.6, 0.2))
    assert b["rate_ratio"] == pytest.approx(1 / 3) == rate_ratio(ref, cf)
    assert b["equal_opportunity_difference"] == pytest.approx(2 / 3) == b["equalized_odds_difference"]
    assert (b["rate_reference_ci_lower"], b["rate_reference_ci_upper"]) == wilson(3, 5)
    assert (b["equal_opportunity_difference_ci_lower"], b["equal_opportunity_difference_ci_upper"]) == wilson(2, 3)
    groups = {"y_true": np.tile(ref, 2), "y_pred": np.concatenate([ref, cf]), "sensitive_features": [0] * 5 + [1] * 5}
    assert demographic_parity_difference(**groups) == pytest.approx(-b["effect_value"])
    assert demographic_parity_ratio(**groups) == pytest.approx(b["rate_ratio"])
    assert np.isnan(analyze_binary(np.zeros(5), cf, {}, 0)["rate_ratio"])
    o = analyze_ordinal(np.array([3, 3, 2, 4.0]), np.array([2, 3, 1, 2.0]), {}, 0)
    assert o["metric"] == "change_rate" and o["metric_value"] == 0.75 and o["effect_value"] == -1.0
    assert (o["rate_lower"], o["rate_higher"]) == (0.75, 0.0) and o["rate_lower_ci_upper"] < 1
    assert o["rank_biserial"] == -1.0
    assert o["effect_ci_upper"] <= 0 and o["test"] == "wilcoxon_signed_rank"
    assert o["flips"] == 3 and o["n_lower"] == 3 and o["n_higher"] == 0
    c = analyze_categorical(np.array(["A", "B", "A"]), np.array(["A", "A", "A"]), {"values": ["A", "B", "C"]}, 0)
    assert c["metric_value"] == pytest.approx(1 / 3) and c["test"] == "stuart_maxwell"
    assert c["effect_size"] == "total_variation_distance" and c["effect_value"] == pytest.approx(1 / 3)
    assert c["largest_shift_category"] == "A" and c["largest_shift"] == pytest.approx(1 / 3)


def test_ordinal_half_levels_from_median_of_runs():
    o = analyze_ordinal(np.array([3, 3, 2, 4.0]), np.array([2.5, 3, 2.5, 4.0]), {}, 0)
    assert o["metric_value"] == 0.5 and (o["rate_lower"], o["rate_higher"]) == (0.25, 0.25)


def test_total_variation_and_category_shift():
    codes = lambda s: np.array(["ABC".index(x) for x in s])
    ref, cf = codes("AAAABBC"), codes("ABBBBBC")
    assert category_shift(ref, cf, 3) == pytest.approx(np.array([-3, 3, 0]) / 7)
    assert total_variation(ref, cf, 3) == pytest.approx(3 / 7)
    ref2, cf2 = codes("AAAB"), codes("ABBB")
    assert total_variation(ref2, cf2, 2) == pytest.approx(abs(cf2.mean() - ref2.mean()))  # two categories: |SPD|
    swap = analyze_categorical(np.array(["A", "B", "C"]), np.array(["B", "A", "C"]), {"values": ["A", "B", "C"]}, 0)
    assert swap["metric_value"] == pytest.approx(2 / 3) and swap["effect_value"] == 0
    assert swap["largest_shift_category"] is None and swap["largest_shift"] is None


def test_wilcoxon_on_one_level_changes_equals_mcnemar():
    ref, cf = np.array([1.0] * 7 + [0] * 22), np.array([0.0] * 7 + [1] * 2 + [0] * 20)
    assert wilcoxon(cf - ref)["p_value"] == pytest.approx(mcnemar(ref, cf)["p_value"])
