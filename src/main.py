import argparse
import importlib.metadata
import json
import platform
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.config import Config, load_config
from src.evaluation.analyze import ALPHA, run_tests
from src.evaluation.dataset import aggregate, analysis_dataset, encode, make_pairs, variables
from src.generation.design import comparisons, conditions
from src.generation.loader import load_cases
from src.generation.parsing import output_model, parse_response
from src.generation.runner import generate_outputs
from src.reporting.figures import FigureData, make_figures
from src.reporting.report import write_report
from src.store import ResultStore


@dataclass
class Context:
    store: ResultStore
    config: Config | None = None
    reparse: bool = False
    manifest: dict = field(default_factory=dict)
    cases: list = field(default_factory=list)
    records: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    pairs: object = None


def _say(stage: str, message: str) -> None:
    print(f"{stage:<10} {message}", flush=True)


def _packages() -> dict:
    lines = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    names = [line.split("==")[0] for line in lines if line and not line.startswith("-")]
    return {name: importlib.metadata.version(name) for name in names}


def new_manifest(cfg: Config, experiment_id: str) -> dict:
    prompts = {k: Path(getattr(cfg.prompts, k)).read_text(encoding="utf-8") for k in ("system", "user")}
    return {"experiment_id": experiment_id, "created": datetime.now().isoformat(timespec="seconds"), "finished": None,
            "config": cfg.model_dump(mode="json", by_alias=True), "prompts": prompts,
            "schema": json.loads(Path(cfg.schema_file).read_text(encoding="utf-8")),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": _packages()}}


def design(ctx: Context) -> None:
    cfg, store = ctx.config, ctx.store
    ctx.cases = load_cases(cfg.dataset, cfg.generation.seed)
    ctx.manifest = new_manifest(cfg, store.root.name)
    store.create()
    store.write_manifest(ctx.manifest)
    n_cond, runs, n_models = len(conditions(cfg)), cfg.generation.repetitions, len(cfg.models)
    _say("design", f"{len(ctx.cases)} cases x {n_cond} conditions x {runs} runs x {n_models} model(s) = "
                  f"{len(ctx.cases) * n_cond * runs * n_models} generations | {len(comparisons(cfg))} comparisons")


def run_generations(ctx: Context) -> None:
    generate_outputs(ctx.config, ctx.cases, ctx.store, ctx.manifest)
    ctx.manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    ctx.store.write_manifest(ctx.manifest)


def load(ctx: Context) -> None:
    ctx.manifest = ctx.store.read_manifest()
    ctx.config = Config.model_validate(ctx.manifest["config"])


def parse(ctx: Context) -> None:
    ctx.records = ctx.store.read_generations()
    if ctx.reparse:
        schema = ctx.manifest["schema"]
        model = output_model(schema)
        for r in tqdm(ctx.records, desc="parse      answers", unit="answer", leave=False):
            if r["raw_response"] is not None:
                r["status"], r["output"], r["message"] = parse_response(r["raw_response"], schema, model)
        ctx.store.write_generations(ctx.records)
    counts = Counter(r["status"] for r in ctx.records).most_common()
    _say("parse", " | ".join(f"{n} {s}" for s, n in counts))


def analyze(ctx: Context) -> None:
    cfg, schema = ctx.config, ctx.manifest["schema"]
    vars_ = variables(schema)
    df = analysis_dataset(ctx.records, ctx.manifest)
    agg = aggregate(encode(df, vars_), vars_)
    ctx.pairs = make_pairs(agg, comparisons(cfg))
    tests = run_tests(ctx.pairs, vars_, cfg, df["case_id"].nunique())
    ctx.tables = {"generations": df, "tests": tests}
    _say("analyze", f"{tests['p_value'].notna().sum()} tests | {int(tests['statistically_significant'].sum())} significant "
                   f"(benjamini_hochberg, alpha={ALPHA})")


def figures(ctx: Context) -> None:
    shutil.rmtree(ctx.store.figures, ignore_errors=True) # če že obstaja in se dela ponovna analiza
    data = FigureData(ctx.tables["tests"], ctx.pairs, ctx.manifest["schema"], ctx.config)
    count = make_figures(data, ctx.store.figures) if ctx.config.visualization.enabled else 0
    _say("figures", f"{count} figures -> {ctx.store.figures}")


def save(ctx: Context) -> None:
    ctx.store.write_tables(ctx.tables)
    _say("results", str(ctx.store.results))


def report(ctx: Context) -> None:
    write_report(ctx.store)
    _say("report", str(ctx.store.report))


PIPELINES = {
    "run": [design, run_generations, parse, analyze, figures, save, report],
    "analyze": [load, parse, analyze, figures, save],
    "report": [report],
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.main")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate outputs, then analyze and report")
    run.add_argument("--config", required=True)
    analyze_ = sub.add_parser("analyze", help="statistics and figures from stored results")
    analyze_.add_argument("--results", required=True)
    analyze_.add_argument("--reparse", action="store_true")
    report_ = sub.add_parser("report", help="HTML report from stored analysis")
    report_.add_argument("--results", required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        ctx = Context(ResultStore(Path(cfg.results_dir) / f"{cfg.name}_{datetime.now():%Y%m%d_%H%M%S}"), cfg)
    else:
        ctx = Context(ResultStore(args.results), reparse=getattr(args, "reparse", False))
    for stage in PIPELINES[args.command]:
        stage(ctx)


if __name__ == "__main__":
    main()
