import pandas as pd

KEYS = ["model", "case_id", "condition"]
ENCODE = {
    "binary": lambda v, var: int(v),
    "ordinal": lambda v, var: var["values"].index(v) + 1,
    "categorical": lambda v, var: v,
    "continuous": lambda v, var: float(v),
}


def variables(schema: dict) -> list[dict]:
    return [{"name": name, "type": f["type"], "values": f.get("values")}
            for name, f in schema["fields"].items() if f["type"] != "free_text"]


def analysis_dataset(records: list[dict], manifest: dict) -> pd.DataFrame:
    df = pd.DataFrame(records)
    outputs = pd.DataFrame([r or {} for r in df["output"]], index=df.index, columns=list(manifest["schema"]["fields"]))
    return pd.concat([df.drop(columns=["output", "raw_response", "message"]), outputs], axis=1)


def encode(df: pd.DataFrame, variables: list[dict]) -> pd.DataFrame:
    valid = df[df["status"] == "VALID"]
    cols = {v["name"]: [ENCODE[v["type"]](x, v) for x in valid[v["name"]]] for v in variables}
    return pd.DataFrame(cols, index=valid.index).assign(**{k: valid[k] for k in KEYS})


def majority(s: pd.Series):
    counts = s.value_counts()
    return counts.index[0] if len(counts) == 1 or counts.iloc[0] > counts.iloc[1] else None


AGGREGATE = {"binary": majority, "ordinal": "median", "categorical": majority, "continuous": "mean"}


def aggregate(encoded: pd.DataFrame, variables: list[dict]) -> pd.DataFrame:
    return encoded.groupby(KEYS).agg({v["name"]: AGGREGATE[v["type"]] for v in variables})


def make_pairs(agg: pd.DataFrame, comparisons: list) -> pd.DataFrame:
    frames = []
    cond = agg.index.get_level_values("condition")
    for comp in comparisons:
        ref = agg[cond == comp.reference].droplevel("condition")
        cf = agg[cond == comp.counterfactual].droplevel("condition")
        j = ref.join(cf, lsuffix="__ref", rsuffix="__cf", how="inner").reset_index()
        j.insert(1, "sensitive_attribute", comp.sensitive_attribute)
        j.insert(2, "comparison", comp.label)
        frames.append(j)
    return pd.concat(frames, ignore_index=True)
