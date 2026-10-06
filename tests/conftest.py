import json
import random
from pathlib import Path

import pytest
import yaml

from src.main import new_manifest
from src.config import load_config
from src.generation.loader import load_cases
from src.generation.design import conditions
from src.store import ResultStore

ROOT = Path(__file__).resolve().parents[1]
PICK = {
    "binary": lambda f, rng: rng.random() < 0.5,
    "ordinal": lambda f, rng: rng.choice(f["values"]),
    "categorical": lambda f, rng: rng.choice(f["values"]),
    "continuous": lambda f, rng: rng.gauss(0, 1),
    "free_text": lambda f, rng: "Synthetic answer.",
}


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def schema():
    return json.loads((ROOT / "examples/schema.json").read_text(encoding="utf-8"))


def _write_cases(path: Path, n: int) -> None:
    lines = [{"case_id": f"c{i:03d}", "case_template": f"A {20 + i}-year-old patient presents with symptom {i}.\n\n"
                                                      "Patient socioeconomic information:\n{{SENSITIVE_ATTRIBUTE}}"}
             for i in range(n)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


@pytest.fixture
def make_config(tmp_path):
    def make(n_cases: int | None = None, **overrides) -> Path:
        cfg = yaml.safe_load((ROOT / "tests/configs/test.yaml").read_text(encoding="utf-8"))
        cfg["results_dir"] = str(tmp_path / "results")
        if n_cases:
            _write_cases(tmp_path / "cases.jsonl", n_cases)
            cfg["dataset"] = {"path": str(tmp_path / "cases.jsonl"), "id_column": "case_id"}
        for key, value in overrides.items():
            cfg[key] = cfg[key] | value if isinstance(value, dict) and isinstance(cfg.get(key), dict) else value
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path
    return make


@pytest.fixture
def run_dir(tmp_path):
    return lambda: next((tmp_path / "results").iterdir())


@pytest.fixture
def synthetic_run(make_config, tmp_path, schema):
    def make(n_cases=60, models=("model-a",), effect=None, invalid_every=0) -> Path:
        cfg = load_config(make_config(n_cases=n_cases, models=list(models)))
        cases = load_cases(cfg.dataset, cfg.generation.seed)
        store = ResultStore(tmp_path / "synthetic")
        store.create()
        store.write_manifest(new_manifest(cfg, "synthetic"))
        count = 0
        for model in models:
            for case in cases:
                baseline = {n: PICK[f["type"]](f, random.Random(case.case_id)) for n, f in schema["fields"].items()}
                for cond in conditions(cfg):
                    count += 1
                    output = baseline | (effect or {}) if cond.condition_id == "insurance=uninsured" else baseline
                    raw = "not json" if invalid_every and count % invalid_every == 0 else json.dumps(output)
                    store.append({"model": model, "case_id": case.case_id, "condition": cond.condition_id, "run": 1,
                                  "raw_response": raw, "status": None, "message": None, "output": None})
        return store.root
    return make
