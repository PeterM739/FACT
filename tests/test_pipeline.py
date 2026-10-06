import json

import pandas as pd
import pytest

from src.main import main
from src.reporting.report import report_data
from src.store import ResultStore

EFFECT = {"urgency": 1, "specialist_referral": False, "recommended_care": "SELF_CARE"}


def test_end_to_end_with_small_model(make_config, run_dir, monkeypatch):
    main(["run", "--config", str(make_config())])
    out = run_dir()
    assert sorted(p.name for p in out.iterdir()) == ["figures", "generations.jsonl", "manifest.yaml", "report.pdf", "results.xlsx"]
    manifest = ResultStore(out).read_manifest()
    assert not {"dataset", "models"} & set(manifest) and manifest["config"]["models"][0]["dtype"]

    records = [json.loads(line) for line in (out / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2 * 6
    assert all(isinstance(r["raw_response"], str) and r["status"] for r in records)
    sheets = pd.read_excel(out / "results.xlsx", sheet_name=None)
    assert set(sheets) == {"generations", "tests"}
    assert list(sheets["generations"].columns[:5]) == ["model", "case_id", "condition", "run", "status"]
    assert "message" not in sheets["generations"]

    monkeypatch.setattr("src.generation.runner.build_model", lambda spec: pytest.fail("analysis must not call a model"))
    before = (out / "manifest.yaml").read_text(encoding="utf-8")
    main(["analyze", "--results", str(out), "--reparse"])
    main(["report", "--results", str(out)])
    assert (out / "manifest.yaml").read_text(encoding="utf-8") == before
    with pytest.raises(FileExistsError):
        ResultStore(out).create()
    assert (out / "report.pdf").read_bytes().startswith(b"%PDF")


def test_injected_effect_is_detected_and_null_is_not(synthetic_run):
    out = synthetic_run(effect=EFFECT, invalid_every=37)
    main(["analyze", "--results", str(out), "--reparse"])
    tests = pd.read_excel(out / "results.xlsx", sheet_name="tests").set_index(["comparison", "variable"])
    assert tests.loc[("original->uninsured", "urgency"), "statistically_significant"]
    assert tests.loc[("original->uninsured", "urgency"), "effect_value"] < 0
    assert tests.loc[("original->uninsured", "specialist_referral"), "statistically_significant"]
    assert not tests.loc[("original->private", "urgency"), "statistically_significant"]
    assert not tests.xs("original->low")["statistically_significant"].any()
    generations = pd.read_excel(out / "results.xlsx", sheet_name="generations")
    assert (generations["status"] == "INVALID_JSON").sum() > 0


def test_figures_follow_data_types_and_models(synthetic_run):
    out = synthetic_run(models=("model-a", "model-b"), effect=EFFECT | {"diagnostic_workup": "ADVANCED"})
    main(["analyze", "--results", str(out), "--reparse"])
    main(["report", "--results", str(out)])
    name = lambda p: f"{p.parent.name}/{p.stem.split('_', 1)[1]}"
    files = {name(p) for p in (out / "figures").rglob("*.png")}
    assert files == {name(p) for p in (out / "figures").rglob("*.pdf")}
    assert {"overview/model-a_overview", "overview/model-b_overview", "ordinal/urgency", "binary/specialist_referral",
            "ordinal/diagnostic_workup", "categorical/recommended_care", "categorical/recommended_care_transitions"} <= files
    assert {f.split("/")[0] for f in files} == {"overview", "binary", "ordinal", "categorical"}
    assert "ordinal/risk_level" not in files

    data = report_data(ResultStore(out))
    outputs = [o for s in data["sections"].values() for o in s]
    embedded = [f["figure"] for f in data["overview"]] + [o[k] for o in outputs for k in ("figure", "transitions") if o[k]]
    assert all(f.endswith(".pdf") for f in embedded) and {name(out / f.lstrip("/")) for f in embedded} == files
    columns = {o["name"]: o["table"]["columns"] for o in outputs}
    assert columns["specialist referral"] == ["comparison", "n", "flip rate", "rate ref.", "SPD", "adj. p"]
    assert columns["urgency"] == ["comparison", "n", "change rate", "mean Δ", "adj. p"]
    assert columns["recommended care"] == ["comparison", "n", "change rate", "TVD", "adj. p"]
    assert data["significant"]["columns"] == ["output", "comparison", "effect", "adj. p"]
    tables = [o["table"] for o in outputs] + [data["significant"]]
    assert all([g["name"] for g in t["groups"]] == ["model-a", "model-b"] and "test" not in t["columns"] for t in tables)
    cells = [c for t in tables for g in t["groups"] for row in g["rows"] for c in row]
    assert "<0.001 ***" in cells and not any("%" in c for c in cells)
    assert "risk level" in data["unchanged"]
