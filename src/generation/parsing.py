import json
import re
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError, create_model

FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
ERROR_STATUS = {"missing": "MISSING_FIELD", "literal_error": "INVALID_ENUM"}
ANNOTATIONS = {
    "binary": lambda f: bool,
    "ordinal": lambda f: Literal[tuple(f["values"])],
    "categorical": lambda f: Literal[tuple(f["values"])],
    "continuous": lambda f: float,
    "free_text": lambda f: str,
}


def output_model(schema: dict):
    fields = {name: (ANNOTATIONS[f["type"]](f), Field(description=f.get("description"))) for name, f in schema["fields"].items()}
    return create_model(schema["name"], __config__=ConfigDict(extra="forbid", strict=True), **fields)


def schema_text(schema: dict) -> str:
    return json.dumps(output_model(schema).model_json_schema(), indent=2)


def parse_response(raw: str, schema: dict, model=None) -> tuple[str, dict | None, str | None]:
    text = raw.strip()
    fenced = FENCE.match(text)
    try:
        data = json.loads(fenced.group(1) if fenced else text)
    except json.JSONDecodeError as e:
        return "INVALID_JSON", None, str(e)
    try:
        output = (model or output_model(schema)).model_validate(data).model_dump()
    except ValidationError as e:
        kinds = {err["type"] for err in e.errors()}
        status = next((s for kind, s in ERROR_STATUS.items() if kind in kinds), "SCHEMA_ERROR")
        return status, None, "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
    return "VALID", output, None
