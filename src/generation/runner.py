import itertools

from tqdm import tqdm

from src.generation.design import conditions, render_case
from src.generation.parsing import output_model, parse_response, schema_text
from src.generation.prompts import build_messages
from src.models.base import build_model


def generate_outputs(cfg, cases, store, manifest) -> None:
    g = cfg.generation
    schema, prompts = manifest["schema"], manifest["prompts"]
    model_cls, text = output_model(schema), schema_text(schema)
    for spec, entry in zip(cfg.models, manifest["config"]["models"]):
        todo = [(run, cond, case) for run in range(1, g.repetitions + 1) for cond in conditions(cfg) for case in cases]
        model = build_model(spec)
        entry["dtype"] = model.dtype
        manifest["environment"] |= model.environment
        store.write_manifest(manifest)
        todo.sort(key=lambda t: (t[0], t[1].seed_offset))
        bar = tqdm(total=len(todo), desc=f"generate   {spec.name}")
        for (run, offset), group in itertools.groupby(todo, key=lambda t: (t[0], t[1].seed_offset)):
            group = list(group)
            for i in range(0, len(group), spec.batch_size):
                batch = group[i:i + spec.batch_size]
                messages = [build_messages(prompts["system"], prompts["user"], render_case(case, cond), text) for _, cond, case in batch]
                try:
                    responses, error = model.generate(messages, g.temperature, g.max_new_tokens, g.seed + run + offset), None
                except Exception as e:
                    responses, error = [None] * len(batch), repr(e)
                for (_, cond, case), raw in zip(batch, responses):
                    status, output, message = parse_response(raw, schema, model_cls) if raw is not None else ("GENERATION_ERROR", None, error)
                    store.append({"model": spec.name, "case_id": case.case_id, "condition": cond.condition_id, "run": run,
                                  "raw_response": raw, "status": status, "message": message, "output": output})
                bar.update(len(batch))
        bar.close()
        model.close()
