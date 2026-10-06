import json
from pathlib import Path

import pandas as pd
import typst

from src.evaluation.analyze import ALPHA
from src.evaluation.metrics import LEVEL, RESAMPLES
from src.evaluation.statistical_tests import PERMUTATIONS
from src.reporting.figures import METRICS, interval, labelled, pretty, result, safe, unchanged

TEMPLATE = Path(__file__).with_name("report.typ")
FORMATS = ("pdf", "svg", "png")

EFFECTS = {"paired_proportion_difference": ("SPD", "+.3f"), "mean_counterfactual_difference": ("mean Δ", "+.3f"),
           "total_variation_distance": ("TVD", ".3f"), "mean_difference": ("mean Δ", "+.3g")}

COLUMNS = {
    "binary": [("rate ref.", "rate_reference", None, ".3f"), ("SPD", "effect_value", "effect", "+.3f")],
    "ordinal": [("mean Δ", "effect_value", "effect", "+.3f")],
    "categorical": [("TVD", "effect_value", "effect", ".3f")],
    "continuous": [("mean Δ", "effect_value", "effect", "+.3g"), ("d_z", "d_z", None, "+.2f")],
}


def _text(v) -> str:
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def _table(df: pd.DataFrame, text: int, by: str | None = None) -> dict:
    """Column names, number of leading text columns and row groups (one per value of `by`)."""
    number = lambda v: _text(v).replace("-", "−")
    groups = df.groupby(by, sort=False) if by else [(None, df)]
    return {"columns": [c for c in df.columns if c != by], "text": text,
            "groups": [{"name": name, "rows": [[number(v) if j >= text else _text(v) for j, v in enumerate(row)]
                                               for row in g.drop(columns=by or []).itertuples(index=False)]} for name, g in groups]}


def _short(models: pd.Series) -> pd.Series:
    short = models.str.split("/").str[-1]
    return short if short.nunique() == models.nunique() else models


def _effect(r) -> str | None:
    """Effect size named in the cell, e.g. 'SPD −0.102 [−0.163, −0.051]'."""
    name, fmt = EFFECTS.get(r.effect_size, (None, None))
    value = fmt and interval(r.effect_value, r.effect_ci_lower, r.effect_ci_upper, fmt)
    return value and f"{name} {value}"


def _column(t: pd.DataFrame, value: str, ci: str | None, fmt: str) -> list:
    lo, hi = (t[f"{ci}_ci_lower"], t[f"{ci}_ci_upper"]) if ci else ([None] * len(t),) * 2
    return [interval(*v, fmt) for v in zip(t[value], lo, hi)]


def _output_table(t: pd.DataFrame) -> dict:
    results = [result(r) for r in t.itertuples()]
    df = pd.DataFrame({"model": _short(t["model"]), "comparison": t["row"], "n": t["n_pairs"],
                       METRICS[t["metric"].dropna().iloc[0]][0]: [m for m, _ in results]})
    for header, value, ci, fmt in COLUMNS[t["variable_type"].iloc[0]]:
        df[header] = _column(t, value, ci, fmt)
    df["adj. p"] = [p for _, p in results]
    return _table(df, 1, "model")


def _significant_table(tests: pd.DataFrame) -> dict:
    s = labelled(tests[tests["statistically_significant"] == True])
    df = pd.DataFrame([(r.model, pretty(r.variable), r.row, _effect(r), result(r)[1]) for r in s.itertuples()],
                      columns=["model", "output", "comparison", "effect", "adj. p"])
    return _table(df.assign(model=_short(df["model"])), 2, "model")


def _outputs(store, tests: pd.DataFrame) -> tuple[list, dict]:
    fmt = next((f for f in FORMATS if any(store.figures.glob(f"*/*.{f}"))), FORMATS[0])
    by_variable = {safe(v): t for v, t in labelled(tests).groupby("variable", sort=False)}
    models = {safe(m, "overview"): m for m in tests["model"].unique()}
    overview, outputs = [], {}
    for p in sorted(store.figures.glob(f"*/*.{fmt}"), key=lambda p: p.name):
        path, key = "/" + p.relative_to(store.root).as_posix(), p.stem.split("_", 1)[1]
        if key in models:
            overview.append({"name": models[key], "figure": path})
        elif key in by_variable:
            t = by_variable[key]
            outputs[key] = {"type": p.parent.name, "name": pretty(t["variable"].iloc[0]), "figure": path, "table": _output_table(t), "transitions": None}
        elif key.removesuffix("_transitions") in outputs:
            outputs[key.removesuffix("_transitions")]["transitions"] = path
    sections = {}
    for o in outputs.values():
        sections.setdefault(o.pop("type"), []).append(o)
    return overview, sections


def report_data(store) -> dict:
    m, tests = store.read_manifest(), pd.read_excel(store.results, sheet_name="tests")
    g = pd.DataFrame(store.read_generations(), columns=["model", "case_id", "status"])
    cfg, gen = m["config"], m["config"]["generation"]
    valid = (g["status"] == "VALID").groupby(g["model"]).mean()
    models = pd.DataFrame([{"model": s["name"], "dtype": s["dtype"], "quantization": s["quantization"] or "none",
                            "GPU": m["environment"].get("gpu"), "valid outputs": f"{valid.get(s['name'], float('nan')):.3f}"}
                           for s in cfg["models"]])
    attributes = pd.DataFrame([(a, v, text) for a, spec in cfg["sensitive_attributes"].items() for v, text in spec["values"].items()], columns=["sensitive attribute", "value", "inserted text"])
    attributes.loc[attributes["sensitive attribute"].duplicated(), "sensitive attribute"] = ""
    fields = pd.DataFrame([(n, f["type"], ", ".join(map(str, f.get("values", [])))) for n, f in m["schema"]["fields"].items()], columns=["field", "type", "values"])
    overview, sections = _outputs(store, tests)
    return {
        "title": m["experiment_id"], "created": _text(m["created"]), "finished": _text(m.get("finished")),
        "dataset": cfg["dataset"]["path"], "cases": int(g["case_id"].nunique()),
        "generation": {k: _text(gen[k]) for k in ("repetitions", "temperature", "max_new_tokens")},
        "counts": ", ".join(f"{k} {v}" for k, v in g["status"].value_counts().items()), "generations": len(g),
        "schema": f"{m['schema']['name']} v{m['schema']['version']}",
        "models": _table(models, 5), "sensitive_attributes": _table(attributes, 3), "fields": _table(fields, 3),
        "statistics": {"alpha": _text(ALPHA), "level": f"{LEVEL:.0%}", "bootstrap": f"{RESAMPLES:,}",
                       "permutations": f"{PERMUTATIONS:,}"},
        "overview": overview, "sections": sections, "tests": int(tests["p_value"].notna().sum()),
        "significant": _significant_table(tests),
        "unchanged": [pretty(v) for v, t in tests.groupby("variable", sort=False) if unchanged(t)],
        "prompts": m["prompts"],
    }


def write_report(store) -> None:
    typst.compile(TEMPLATE.read_bytes(), output=str(store.report), root=str(store.root), ignore_system_fonts=True,
                  sys_inputs={"data": json.dumps(report_data(store), ensure_ascii=False)})
