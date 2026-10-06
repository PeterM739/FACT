from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RESERVED_CONDITIONS = {"original", "retest"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetConfig(Strict):
    path: str
    sheet: str | int | None = None
    text_column: str = "case_template"
    id_column: str | None = None
    metadata_columns: list[str] | None = None
    placeholder: str = "{{SENSITIVE_ATTRIBUTE}}"
    limit: int | None = None
    sample: int | None = None


class SensitiveAttribute(Strict):
    values: dict[str, str]
    reference: str = "original"
    comparisons: list[tuple[str, str]] | Literal["all_pairs"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _short_form(cls, data):
        return {"values": data} if isinstance(data, dict) and "values" not in data else data

    @model_validator(mode="after")
    def _check(self):
        if RESERVED_CONDITIONS & set(self.values):
            raise ValueError(f"value names {sorted(RESERVED_CONDITIONS)} are reserved")
        known = set(self.values) | {"original"}
        used = {self.reference} | {v for pair in (self.comparisons if isinstance(self.comparisons, list) else []) for v in pair}
        if used - known:
            raise ValueError(f"unknown sensitive attribute values: {sorted(used - known)}")
        return self


class ModelSpec(Strict):
    name: str
    provider: str = "transformers"
    dtype: str | None = None
    quantization: Literal["4bit", "8bit"] | None = None
    device: Literal["auto", "cpu"] = "auto"
    batch_size: int = 8

    @model_validator(mode="before")
    @classmethod
    def _short_form(cls, data):
        return {"name": data} if isinstance(data, str) else data


class Prompts(Strict):
    system: str = "configs/prompts/system.txt"
    user: str = "configs/prompts/user.txt"


class Generation(Strict):
    temperature: float = 0.0
    max_new_tokens: int = 512
    repetitions: int = 1
    seed: int = 42


class Visualization(Strict):
    enabled: bool = True
    formats: list[str] = ["pdf", "png"]
    dpi: int = 300


class Config(Strict):
    name: str
    dataset: DatasetConfig
    schema_file: str = Field(alias="schema")
    models: list[ModelSpec]
    sensitive_attributes: dict[str, SensitiveAttribute]
    original_value: str = ""
    prompts: Prompts = Prompts()
    generation: Generation = Generation()
    visualization: Visualization = Visualization()
    results_dir: str = "results"


def load_config(path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
