import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Case:
    case_id: str
    case_template: str
    placeholder: str
    metadata: dict = field(default_factory=dict)


READERS = {
    ".jsonl": lambda p, cfg: pd.read_json(p, lines=True, dtype=False),
    ".json": lambda p, cfg: pd.read_json(p, dtype=False),
    ".csv": lambda p, cfg: pd.read_csv(p),
    ".tsv": lambda p, cfg: pd.read_csv(p, sep="\t"),
    ".xlsx": lambda p, cfg: pd.read_excel(p, sheet_name=cfg.sheet or 0),
    ".parquet": lambda p, cfg: pd.read_parquet(p),
}


def _resolve_path(cfg) -> Path:
    if not cfg.path.startswith("hf://datasets/"):
        return Path(cfg.path)
    from huggingface_hub import hf_hub_download

    owner, repo, filename = cfg.path.removeprefix("hf://datasets/").split("/", 2)
    return Path(hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset"))


def _load_table(cfg,path: Path) -> list[Case]:
    df = READERS[path.suffix.lower()](path, cfg).dropna(subset=[cfg.text_column])
    ids = df[cfg.id_column].astype(str) if cfg.id_column else [f"case_{i + 1:04d}" for i in df.index]
    meta = cfg.metadata_columns if cfg.metadata_columns is not None else [c for c in df.columns if c not in (cfg.text_column, cfg.id_column)]
    return [Case(case_id, str(row[cfg.text_column]), cfg.placeholder, row[meta].to_dict()) for case_id, (_, row) in zip(ids, df.iterrows())]


def load_cases(cfg, seed: int) -> list[Case]:
    cases = _load_table(cfg,_resolve_path(cfg))
    missing = [c.case_id for c in cases if cfg.placeholder not in c.case_template]
    if missing:
        raise ValueError(f"{len(missing)} cases do not contain the placeholder {cfg.placeholder!r}: {missing[:10]}")
    if cfg.sample:
        keep = set(random.Random(seed).sample(range(len(cases)), cfg.sample))
        cases = [c for i, c in enumerate(cases) if i in keep]
    return cases[:cfg.limit] if cfg.limit else cases
