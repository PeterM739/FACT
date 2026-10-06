import importlib
from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    dtype: str | None = None
    environment: dict = {}

    @abstractmethod
    def generate(self, batch: list[list[dict]], temperature: float, max_new_tokens: int, seed: int) -> list[str]: ...

    def close(self) -> None:
        pass


MODELS = {
    "transformers": "src.models.transformers_adapter:TransformersAdapter",
}


def build_model(spec) -> ModelAdapter:
    module, cls = MODELS[spec.provider].split(":")
    return getattr(importlib.import_module(module), cls)(spec)
