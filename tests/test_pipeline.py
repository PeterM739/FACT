import json

import pandas as pd
import pytest

from src.config import Config
from src.main import main
from src.reporting.report import report_data
from src.store import ResultStore

EFFECT = {"urgency": 1, "specialist_referral": False, "recommended_care": "SELF_CARE"}


def _store(out) -> ResultStore:
    return ResultStore(out, Config.model_validate(ResultStore(out).read_manifest()["config"]).analysis)


def test_end_to_end_with_small_model(make_config, run_dir, monkeypatch):
    main(["run", "--config", str(make_config())])
    out = run_dir()
    store = _store(out)
    assert sorted(p.name for p in out.iterdir()) == [store.out.name, "generations.jsonl", "manifest.yaml"]
    assert sorted(p.name for p in store.out.iterdir()) == ["figures", "report.pdf", "results.xlsx"]
    manifest = store.read_manifest()
    assert not {"dataset", "models"} & set(manifest) and manifest["config"]["models"][0]["dtype"]

    records = [json.loads(line) for line in (out / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2 * 6
    assert all(isinstance(r["raw_response"], str) and r["status"] for r in records)
    sheets = pd.read_excel(store.results, sheet_name=None)
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
    assert store.report.read_bytes().startswith(b"%PDF")


def test_injected_effect_is_detected_and_null_is_not(synthetic_run):
    out = synthetic_run(effect=EFFECT, invalid_every=37)
    main(["analyze", "--results", str(out), "--reparse"])
    tests = pd.read_excel(_store(out).results, sheet_name="tests").set_index(["comparison", "variable"])
    assert tests.loc[("original->uninsured", "urgency"), "statistically_significant"]
    assert tests.loc[("original->uninsured", "urgency"), "effect_value"] < 0
    assert tests.loc[("original->uninsured", "specialist_referral"), "statistically_significant"]
    assert not tests.loc[("original->private", "urgency"), "statistically_significant"]
    assert not tests.xs("original->low")["statistically_significant"].any()
    generations = pd.read_excel(_store(out).results, sheet_name="generations")
    assert (generations["status"] == "INVALID_JSON").sum() > 0


def test_each_setting_gets_its_own_analysis_folder(synthetic_run):
    out = synthetic_run(effect=EFFECT)
    main(["analyze", "--results", str(out), "--reparse"])
    first = _store(out)
    manifest = first.read_manifest()
    manifest["config"]["statistics"] = {"alpha": 1e-12}
    manifest["config"]["visualization"]["formats"] = ["png"]
    first.write_manifest(manifest)
    main(["analyze", "--results", str(out)])
    main(["report", "--results", str(out)])
    second = _store(out)
    assert (first.out.name, second.out.name) == ("alpha_0.05_pdf-png_300dpi", "alpha_1e-12_png_300dpi")
    significant = lambda s: pd.read_excel(s.results, sheet_name="tests")["statistically_significant"]
    assert significant(first).any() and not significant(second).any()
    assert {p.suffix for p in second.figures.rglob("*.*")} == {".png"} and second.report.exists()
    assert report_data(second)["statistics"]["stars"] == "* < 1e-12"


def test_figures_follow_data_types_and_models(synthetic_run):
    out = synthetic_run(models=("model-a", "model-b"), effect=EFFECT | {"diagnostic_workup": "ADVANCED"})
    main(["analyze", "--results", str(out), "--reparse"])
    main(["report", "--results", str(out)])
    store = _store(out)
    name = lambda p: f"{p.parent.name}/{p.stem.split('_', 1)[1]}"
    files = {name(p) for p in store.figures.rglob("*.png")}
    assert files == {name(p) for p in store.figures.rglob("*.pdf")}
    assert {"overview/model-a_overview", "overview/model-b_overview", "ordinal/urgency", "binary/specialist_referral",
            "ordinal/diagnostic_workup", "nominal/recommended_care", "nominal/recommended_care_transitions"} <= files
    assert {f.split("/")[0] for f in files} == {"overview", "binary", "ordinal", "nominal"}
    assert "ordinal/risk_level" not in files

    data = report_data(store)
    outputs = [o for s in data["sections"].values() for o in s]
    embedded = [f["figure"] for f in data["overview"]] + [o[k] for o in outputs for k in ("figure", "transitions") if o[k]]
    assert all(f.endswith(".pdf") for f in embedded) and {name(out / f.lstrip("/")) for f in embedded} == files
    columns = {o["name"]: o["table"]["columns"] for o in outputs}
    assert columns["specialist referral"] == ["comparison", "n", "change rate", "SPD", "DI", "adj. p"]
    assert columns["urgency"] == ["comparison", "n", "change rate", "mean Δ", "mean |Δ|", "adj. p"]
    assert columns["recommended care"] == ["comparison", "n", "change rate", "largest shift", "TVD", "adj. p"]
    care = next(o["table"] for o in outputs if o["name"] == "recommended care")
    assert any(row[3].startswith("self care +") for g in care["groups"] for row in g["rows"])
    assert data["significant"]["columns"] == ["output", "comparison", "effect", "adj. p"]
    tables = [o["table"] for o in outputs] + [data["significant"]]
    assert all([g["name"] for g in t["groups"]] == ["model-a", "model-b"] and "test" not in t["columns"] for t in tables)
    cells = [c for t in tables for g in t["groups"] for row in g["rows"] for c in row]
    assert "<0.001 ***" in cells and not any("%" in c for c in cells)
    assert "risk level" in data["unchanged"]
    assert data["statistics"]["stars"] == "* < 0.05, ** < 0.01, *** < 0.001"
