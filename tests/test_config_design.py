import pytest
from pydantic import ValidationError

from src.config import Config, SensitiveAttribute, load_config
from src.generation.loader import Case
from src.generation.design import RETEST_SEED_OFFSET, comparisons, conditions, render_case

BASE = {"name": "t", "dataset": {"path": "x.jsonl"}, "schema": "s.json", "models": ["some/model"]}


def _config(**sensitive_attributes):
    return Config.model_validate(BASE | {"sensitive_attributes": sensitive_attributes})


def test_sensitive_attribute_short_and_long_form():
    assert SensitiveAttribute.model_validate({"a": "text a", "b": "text b"}).values == {"a": "text a", "b": "text b"}
    long = SensitiveAttribute.model_validate({"values": {"low": "l", "high": "h"}, "comparisons": "all_pairs"})
    assert long.values == {"low": "l", "high": "h"} and long.comparisons == "all_pairs"


@pytest.mark.parametrize("bad", [
    {"original": "x", "b": "y"},
    {"values": {"a": "1"}, "comparisons": [["a", "zzz"]]},
    {"values": {"a": "1"}, "reference": "zzz"},
])
def test_sensitive_attribute_validation(bad):
    with pytest.raises(ValidationError):
        SensitiveAttribute.model_validate(bad)


def test_defaults_and_model_shorthand():
    cfg = load_config("examples/example.yaml")
    assert cfg.models[0].name == "Qwen/Qwen2.5-1.5B-Instruct" and cfg.models[0].batch_size == 8
    assert cfg.generation.temperature == 0.0
    assert cfg.original_value == ""


def test_conditions_and_retest_seed_offset():
    cfg = _config(insurance={"private": "P", "uninsured": "U"})
    conds = conditions(cfg)
    assert [c.condition_id for c in conds] == ["original", "insurance=private", "insurance=uninsured", "retest"]
    assert conds[-1].text == conds[0].text and conds[-1].seed_offset == RETEST_SEED_OFFSET


def test_comparisons():
    cfg = _config(insurance={"values": {"a": "A", "b": "B", "c": "C"}, "comparisons": "all_pairs"},
                 income={"values": {"low": "L", "high": "H"}, "reference": "low"})
    labels = [(c.sensitive_attribute, c.label) for c in comparisons(cfg)]
    assert labels == [("insurance", "original->a"), ("insurance", "original->b"), ("insurance", "original->c"),
                      ("insurance", "a->b"), ("insurance", "a->c"), ("insurance", "b->c"),
                      ("income", "low->high"), ("control", "original->retest")]
    comp = comparisons(cfg)[3]
    assert (comp.reference, comp.counterfactual) == ("insurance=a", "insurance=b")


def test_render_case_keeps_clinical_text():
    case = Case("c1", "A 40-year-old {patient} presents with pain.", "{patient}")
    cfg = _config(insurance={"uninsured": "patient without health insurance"}).model_copy(update={"original_value": "patient"})
    texts = [render_case(case, c) for c in conditions(cfg)]
    assert texts == ["A 40-year-old patient presents with pain.",
                     "A 40-year-old patient without health insurance presents with pain.",
                     "A 40-year-old patient presents with pain."]
