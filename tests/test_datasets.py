import pandas as pd
import pytest

from src.config import DatasetConfig
from src.generation.loader import load_cases


def test_jsonl_dataset():
    cases = load_cases(DatasetConfig(path="examples/cases.jsonl", id_column="case_id"), seed=1)
    assert [c.case_id for c in cases] == ["case_001", "case_002", "case_003"]
    assert all("{{SENSITIVE_ATTRIBUTE}}" in c.case_template for c in cases)
    assert cases[0].metadata == {"specialty": "cardiology"}


def test_excel_dataset_with_inline_placeholder(tmp_path):
    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Case": ["A 30-year-old {patient} presents.", None, "A 50-year-old {patient} faints."],
                      "HR": [80, 90, 100]}).to_excel(writer, sheet_name="Main", index=False)
        pd.DataFrame({"Case": ["ignored"]}).to_excel(writer, sheet_name="Other", index=False)
    cfg = DatasetConfig(path=str(path), sheet="Main", text_column="Case", placeholder="{patient}")
    cases = load_cases(cfg, seed=1)
    assert [c.case_id for c in cases] == ["case_0001", "case_0003"] and cases[1].metadata == {"HR": 100}


def test_missing_placeholder_fails(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"case_template": ["has {{SENSITIVE_ATTRIBUTE}}", "has none"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="case_0002"):
        load_cases(DatasetConfig(path=str(path)), seed=1)


def test_huggingface_path(monkeypatch, tmp_path):
    local = tmp_path / "file.csv"
    pd.DataFrame({"case_template": [f"case {i} {{{{SENSITIVE_ATTRIBUTE}}}}" for i in range(10)]}).to_csv(local, index=False)
    calls = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: calls.append((a, k)) or str(local))
    cfg = DatasetConfig(path="hf://datasets/owner/repo/dir/file.csv", sample=4, limit=3)
    cases = load_cases(cfg, seed=7)
    assert calls == [(("owner/repo", "dir/file.csv"), {"repo_type": "dataset"})]
    assert len(cases) == 3
    assert [c.case_id for c in cases] == [c.case_id for c in load_cases(cfg, seed=7)]
