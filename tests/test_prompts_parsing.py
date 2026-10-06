import json

import pytest

from src.generation.parsing import parse_response, schema_text
from src.generation.prompts import build_messages, render


def _answer(**changes):
    base = {"urgency": 3, "risk_level": 2, "recommended_care": "ROUTINE_PRIMARY_CARE", "diagnostic_workup": "IMAGING",
            "treatment_category": "SUPPORTIVE", "specialist_referral": False, "follow_up": "WITHIN_DAYS",
            "access_adaptation": False, "socioeconomic_factor_used": False, "confidence": 3, "brief_justification": "Short."}
    return json.dumps({k: v for k, v in (base | changes).items() if v is not ...})


def test_render_is_strict():
    assert render("Case: {{CASE}}", CASE="x") == "Case: x"
    with pytest.raises(KeyError, match="SCHEMA"):
        render("{{CASE}} {{SCHEMA}}", CASE="x")


def test_messages(schema):
    messages = build_messages("Be brief.", "Case:\n{{CASE}}\n{{SCHEMA}}", "a patient", schema_text(schema))
    assert messages[0] == {"role": "system", "content": "Be brief."}
    assert messages[1]["role"] == "user" and "a patient" in messages[1]["content"] and '"urgency"' in messages[1]["content"]
    assert [m["role"] for m in build_messages(" \n", "{{CASE}}", "x", "")] == ["user"]


@pytest.mark.parametrize("raw, status", [
    (_answer(), "VALID"),
    ("```json\n" + _answer() + "\n```", "VALID"),
    ("not json", "INVALID_JSON"),
    (_answer(urgency=...), "MISSING_FIELD"),
    (_answer(recommended_care="HOSPITAL"), "INVALID_ENUM"),
    (_answer(urgency=7), "INVALID_ENUM"),
    (_answer(diagnostic_workup="CT"), "INVALID_ENUM"),
    (_answer(specialist_referral="yes"), "SCHEMA_ERROR"),
    (_answer(extra=1), "SCHEMA_ERROR"),
    ("[1, 2]", "SCHEMA_ERROR"),
])
def test_statuses(schema, raw, status):
    assert parse_response(raw, schema)[0] == status
