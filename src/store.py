import json
from pathlib import Path

import pandas as pd
import yaml


def _str(dumper, s):
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)


yaml.add_representer(str, _str, Dumper=yaml.SafeDumper)


class ResultStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = self.root / "manifest.yaml"
        self.generations = self.root / "generations.jsonl"
        self.results = self.root / "results.xlsx"
        self.report = self.root / "report.pdf"
        self.figures = self.root / "figures"

    def create(self) -> None:
        self.root.mkdir(parents=True)

    def write_manifest(self, data: dict) -> None:
        self.manifest.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def read_manifest(self) -> dict:
        return yaml.safe_load(self.manifest.read_text(encoding="utf-8"))

    def append(self, record: dict) -> None:
        with open(self.generations, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_generations(self) -> list[dict]:
        if not self.generations.exists():
            return []
        with open(self.generations, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def write_generations(self, records: list[dict]) -> None:
        self.generations.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    def write_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        with pd.ExcelWriter(self.results) as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name, index=False)
