import re

PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


def render(template: str, **values) -> str:
    missing = set(PLACEHOLDER.findall(template)) - values.keys()
    if missing:
        raise KeyError(f"unresolved prompt placeholders: {sorted(missing)}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def build_messages(system_prompt: str, user_prompt: str, case_text: str, schema_text: str) -> list[dict]:
    values = {"CASE": case_text, "SCHEMA": schema_text}
    messages = [{"role": "user", "content": render(user_prompt, **values)}]
    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": render(system_prompt, **values)})
    return messages
