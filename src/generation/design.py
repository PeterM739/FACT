import itertools
from dataclasses import dataclass

RETEST_SEED_OFFSET = 100_000


@dataclass(frozen=True)
class Condition:
    condition_id: str
    text: str
    seed_offset: int = 0


@dataclass(frozen=True)
class Comparison:
    sensitive_attribute: str
    reference: str
    counterfactual: str
    label: str


def conditions(cfg) -> list[Condition]:
    return [Condition("original", cfg.original_value),
            *(Condition(f"{a}={v}", text) for a, attr in cfg.sensitive_attributes.items() for v, text in attr.values.items()),
            Condition("retest", cfg.original_value, RETEST_SEED_OFFSET)]


def comparisons(cfg) -> list[Comparison]:
    out = []
    for a, attr in cfg.sensitive_attributes.items():
        pairs = [(attr.reference, v) for v in attr.values if v != attr.reference]
        if attr.comparisons == "all_pairs":
            pairs += list(itertools.combinations(attr.values, 2))
        elif attr.comparisons:
            pairs += [tuple(p) for p in attr.comparisons]
        cid = lambda v: "original" if v == "original" else f"{a}={v}"
        out += [Comparison(a, cid(r), cid(c), f"{r}->{c}") for r, c in dict.fromkeys(pairs)]
    return out + [Comparison("control", "original", "retest", "original->retest")]


def render_case(case, cond: Condition) -> str:
    return case.case_template.replace(case.placeholder, cond.text)
