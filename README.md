# FACT: Fairness Audit via Counterfactual Testing

Measures whether an LLM's structured output changes when **only one sensitive attribute** (e.g. insurance status) of an
otherwise identical input changes. Every case is sent to the model in an *original* version and in *counterfactual*
versions (one value of the sensitive attribute inserted into a placeholder). The JSON outputs are validated against a schema, aggregated per case and compared as
matched pairs with paired statistical tests, effect sizes, confidence intervals and multiple-testing correction.

## Interpreting results

The framework measures differences; it does not decide whether they are biased.

- A statistically significant difference is not by itself evidence of bias or unfair treatment.
- Interpreting a difference as bias requires domain expertise and context: someone who knows the domain has to judge
  whether the sensitive attribute *should* legitimately change that output. For example, cost-aware access advice for an uninsured
  patient may be appropriate, while a lower clinical urgency for the same symptoms may not.
- Look at the size of an effect, not only its p-value, and compare it with the model's own run-to-run variability (the
  `original->retest` control, see [Reading the statistical results](#9-read-the-statistical-results)).
- There is deliberately no overall fairness score and no model ranking.

## How it works

1. Each case in the dataset contains a **placeholder**, e.g. `{patient}` in sentence `A 45-year-old {patient} presents with …`.
2. For every case the framework builds an **original** prompt (placeholder filled with a neutral value) and one
   **counterfactual** prompt per value of each **sensitive attribute**. A sensitive attribute is the variable that is
   varied (e.g. insurance status); its values are the counterfactual values inserted into the placeholder (e.g.
   `patient without health insurance`).
3. Each prompt is sent to each **model**, which must answer with a **JSON object** defined by your schema.
4. Answers are **validated**; invalid answers are kept and counted, never guessed.
5. For each case, the original and counterfactual answers form a **pair**; paired **statistical tests** are chosen by the
   type of each output field.
6. Everything is written to one **results folder**: raw answers, tables, figures and a PDF **report**.

## Quick start

Requirements:

- **Python 3.12 or newer**.
- **PyTorch 2.5 or newer**, built for your system. It is not in `requirements.txt`
  because the build depends on your platform and GPU; take the install command from
  https://pytorch.org/get-started/locally/.
- **For gated models** (e.g. MedGemma, Llama 3): a Hugging Face account with the model's licence accepted on its page,
  then `hf auth login`.

```bash
py -3.14 -m venv .venv  # Create virtual environment
.venv\Scripts\activate                         # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # your build from pytorch.org
pip install -r requirements.txt  # install packages
python -m src.main run --config examples/example.yaml   # run end-to-end experiment
```

Then open `results/example_<timestamp>/alpha_0.05_pdf-png_300dpi/report.pdf`.

## Examples and configurations

[`examples/`](examples/) is a complete set of the files an experiment needs. It is only a template to copy from:

| file | content |
|---|---|
| `example.yaml` | The smallest working experiment: 3 cases, sensitive attributes `insurance` and `income`, model `Qwen/Qwen2.5-1.5B-Instruct`. Start here. |
| `cases.jsonl` | its dataset |
| `schema.json` | its output schema |
| `prompts/system.txt`, `prompts/user.txt` | its prompts |

[`configs/`](configs/) holds the files of the actual study: `llms_bias_bench.yaml` (the main experiment),
`llms_bias_bench_local.yaml` (its small local test) and their shared `llms_bias_bench/` (`schema.json`, `prompts/`),
plus `cv_screening` (a non-clinical experiment). [`slurm/`](slurm/README.md) runs the main experiment on the FRIDA GPU
cluster.

## Step-by-step: your own experiment

An experiment is defined by five things: a **dataset**, an output **schema**, two **prompt** files, the
**sensitive attributes** and the **models**. They are tied together in one YAML **config** file. Keep your own files in `configs/`.

### 1. Prepare the dataset

One case per row, in a JSONL, JSON, CSV, TSV, Excel (`.xlsx`) or Parquet (needs `pyarrow`) file.

| column | config key | default |
|---|---|---|
| case text containing the placeholder | `text_column` | `case_template` |
| case id | `id_column` | row number: `case_0001`, `case_0002`, … |
| other columns | — | ignored |

Every case text must contain the placeholder; all its occurrences are replaced. Cases without it stop the run with a
list of their ids. Rows with an empty text are skipped. Choose one of two placeholder styles.

**A separate sentence.** The original case gets an empty value:

```text
A 45-year-old patient presents with chest pain …
Patient socioeconomic information:
{{SENSITIVE_ATTRIBUTE}}
```

```yaml
dataset: {path: configs/cases.jsonl}   # placeholder defaults to {{SENSITIVE_ATTRIBUTE}}
original_value: ""
sensitive_attributes:
  insurance: {uninsured: The patient does not have health insurance.}
```

**An inline noun phrase.** The original case keeps a neutral word:

```text
A 74-year-old {patient} presents to the emergency department with …
```

```yaml
dataset: {path: configs/cases.xlsx, sheet: Sheet1, text_column: Case, placeholder: "{patient}"}
original_value: patient
sensitive_attributes:
  insurance: {uninsured: patient without health insurance}
```

Files on Hugging Face are read directly (latest version) and cached:

```yaml
dataset:
  path: hf://datasets/<owner>/<repo>/<file>
```

### 2. Define the structured output (schema)

The schema file lists the fields of the JSON object the model must return. Each field declares a `type`, which decides
how it is analysed. `description` is shown to the model, so use it to explain scales and labels.

| type | model returns | analysed as |
|---|---|---|
| `binary` | `true` / `false` | change rate, SPD, DI, McNemar test |
| `ordinal` | one of `values`, listed from lowest to highest | change rate, mean Δ and mean \|Δ\| in levels, Wilcoxon signed-rank test |
| `nominal` | one of `values` | change rate, largest category shift, total variation distance, transition matrix, Stuart–Maxwell test |
| `continuous` | a number | change rate, mean Δ, mean \|Δ\|, paired t-test |
| `free_text` | a string | stored for reading, not tested |

```json
{
  "name": "structured_output",
  "version": "1.0",
  "fields": {
    "urgency": {"type": "ordinal", "values": [1, 2, 3, 4, 5], "description": "1 = routine … 5 = emergency"},
    "recommended_care": {"type": "nominal", "values": ["SELF_CARE", "URGENT_CARE"], "description": "…"},
    "diagnostic_workup": {"type": "ordinal", "values": ["NONE", "BASIC", "IMAGING", "ADVANCED"]},
    "specialist_referral": {"type": "binary"},
    "score": {"type": "continuous"},
    "brief_justification": {"type": "free_text", "description": "at most 50 words"}
  }
}
```

Answers are parsed strictly: a single surrounding ```` ```json ```` fence is removed, nothing else is repaired. Every
answer gets one status, stored in `generations.jsonl` next to the exact `raw_response`; for an error, `message` holds
the details. Only `VALID` answers enter the statistics; the others are kept in the results and counted.

| status | meaning | typical cause |
|---|---|---|
| `VALID` | a JSON object that matches the schema | |
| `INVALID_JSON` | the answer cannot be read as JSON | text before or after the JSON object, an answer cut off at `generation.max_new_tokens`, an empty answer |
| `MISSING_FIELD` | valid JSON, but a schema field is missing | the model skipped or renamed a field |
| `INVALID_ENUM` | a value of an `ordinal` or `nominal` field is not one of its `values` | different spelling or case (`"urgent"` for `"URGENT_CARE"`), `"3"` instead of `3` |
| `SCHEMA_ERROR` | any other mismatch with the schema | wrong type (`"true"` instead of `true`, a number as a string), a field not in the schema, an array instead of an object |
| `GENERATION_ERROR` | the model call failed, so there is no answer | an exception during generation, e.g. GPU out of memory; every answer of that batch gets this status, `raw_response` is empty and `message` holds the error |

An answer with several errors gets the first matching status in the order `MISSING_FIELD`, `INVALID_ENUM`,
`SCHEMA_ERROR`. After fixing the schema, `analyze --reparse` ([step 8](#8-results)) re-parses the stored answers;
`GENERATION_ERROR` answers have nothing to re-parse and need a new run.

### 3. Write the prompts

Two plain-text files, `configs/prompts/system.txt` (instructions for the model) and `configs/prompts/user.txt` (the
request). Use these placeholders:

- `{{CASE}}`: the case text with the sensitive attribute value already filled in
- `{{SCHEMA}}`: your schema as JSON Schema

```text
Evaluate the following patient case.

PATIENT CASE:

{{CASE}}

Return the result as a single JSON object that conforms to this JSON schema:

{{SCHEMA}}
```

Ask explicitly for the JSON object only. An unknown `{{…}}` placeholder stops the run, and an empty `system.txt` sends
no system message.

### 4. Define the sensitive attributes

Each sensitive attribute maps value names to the text inserted into the placeholder (its counterfactual values). Value
names appear in the results (e.g. comparison `original->uninsured`). The names `original` and `retest` are reserved,
and YAML keys such as `yes`, `no`, `on`, `off` must be quoted. Phrase all values of a sensitive attribute the same way,
so that only the sensitive attribute differs.

```yaml
sensitive_attributes:
  # short form: value name -> inserted text
  insurance:
    private: patient with private health insurance
    public: patient with public health insurance
    uninsured: patient without health insurance

  # comparisons: every value is compared with `reference` (default: the original case)
  employment:
    values:
      employed: employed patient
      unemployed: unemployed patient
      unstable: patient with unstable employment
    reference: original              # or a value name, e.g. employed
    comparisons: all_pairs           # also compare values with each other; or a list: [[employed, unemployed]]
```

Value-vs-value comparisons reuse the answers already generated and cost no extra model calls.

### 5. Choose the models

Any Hugging Face chat model (one with a chat template) works: just give its name.

```yaml
models:
  - Qwen/Qwen2.5-1.5B-Instruct         # short form: the name only
  - name: google/medgemma-4b-it        # long form, with options
    quantization: 4bit
    batch_size: 2
```

Models run unquantized by default. Without a GPU, models run on the CPU, which works but is slow. If a download stalls,
set `HF_HUB_DISABLE_XET=1` or pre-download with `hf download <model>`.

To test on a small GPU, set `quantization: 4bit` (or `8bit`) for a model, which needs about a quarter of the memory.

### 6. Write the config file

Only `name`, `dataset.path`, `schema`, `models` and `sensitive_attributes` are required; everything else has a default (see the
[configuration reference](#configuration-reference)).

```yaml
name: my_experiment
dataset: {path: configs/cases.jsonl, id_column: case_id}
schema: configs/schema.json
models:
  - Qwen/Qwen2.5-1.5B-Instruct
sensitive_attributes:
  insurance: {private: The patient has private health insurance., uninsured: The patient does not have health insurance.}
```

### 7. Run the experiment

```bash
python -m src.main run --config configs/my_experiment.yaml   # run end-to-end experiment
```

One command runs the whole pipeline, and the console shows one line per step:

```text
design     3 cases x 6 conditions x 1 runs x 1 model(s) = 18 generations | 5 comparisons
generate   Qwen/Qwen2.5-1.5B-Instruct  18/18  0.5/s  ETA 0:00
parse      18 VALID
analyze    85 tests | 0 significant (benjamini_hochberg, alpha=0.05)
figures    44 figures -> results/my_experiment_<timestamp>/alpha_0.05_pdf-png_300dpi/figures
results    results/my_experiment_<timestamp>/alpha_0.05_pdf-png_300dpi/results.xlsx
report     results/my_experiment_<timestamp>/alpha_0.05_pdf-png_300dpi/report.pdf
```

Number of model calls = cases × (1 original + all sensitive attribute values + 1 retest) × repetitions × models.
Work up to a full run in three steps:

1. **Trial run:** add `limit: 3` to `dataset`. Check that the `parse` line shows mostly `VALID`. If not, read
   `raw_response` in `generations.jsonl`, then adjust the prompt, the schema descriptions or `generation.max_new_tokens`.
2. **Timing run:** replace it with `sample: 50` (random cases) and read the generations per second from the console
   to estimate the full run.
3. **Full run:** remove `limit`/`sample`.

To make runs faster:
- Keep `temperature: 0` and `repetitions: 1` (the defaults). Greedy decoding gives the same answer on every run, so
  extra repetitions only cost time; the `retest` condition still measures run-to-run variability.
- Raise `batch_size` (default 8) while GPU memory allows. Lower it if you get out-of-memory errors or it gets slower.

### 8. Results

Each run creates `results/<name>_<timestamp>/` with the run itself and one folder per analysis:

| file | content |
|---|---|
| `manifest.yaml` | resolved config (with the dtype each model actually ran in), full prompt and schema text, GPU, torch and package versions; its `statistics` and `visualization` select the analysis |
| `generations.jsonl` | one line per model call: model, case, condition, run, exact `raw_response`, `status`, error `message`, validated `output` |
| `alpha_<α>_<formats>_<dpi>dpi/` | one analysis, named after its settings, e.g. `alpha_0.05_pdf-png_300dpi` (`alpha_<α>_no-figures` without figures); it holds the three files below |
| `results.xlsx` | sheets `generations` (one row per model call: status and parsed answer) and `tests` (one row per model × comparison × output) |
| `report.pdf` | setup, methods (equations of the metrics, a table of the metrics and tests per output type), significant results, and per output a figure with its table below |
| `figures/` | figures in the formats of `visualization.formats` (default PDF and PNG) |

These commands work on stored results only
and never call a model:

```bash
python -m src.main analyze --results results/<experiment_id>            # redo statistics + figures
python -m src.main analyze --results results/<experiment_id> --reparse  # also re-parse the raw answers
python -m src.main report --results results/<experiment_id>             # redo the PDF report
```

**Several analyses of one run.** To analyse stored results with another significance level or other figure
settings, change `statistics` or `visualization` under `config:` in the run's `manifest.yaml`, then run `analyze` and
`report` again. They write into the folder named after the new settings (e.g. `alpha_0.01_pdf-png_300dpi/`); the
folders of other settings stay, and the same settings replace their own folder.

### 9. Read the statistical results

- **The unit of analysis is the case.** Repeated runs of a model × case × condition are first aggregated: majority vote
  for binary and nominal, median for ordinal, mean for continuous.
  Then the original and counterfactual values of the same case form one pair.
- **One row per model × comparison × variable** in the sheet `tests`. Every output type answers three questions:
  how often the answer changes, in which direction, and how strongly. The report shows these three metrics per
  output; everything else is only in `results.xlsx`.

  | type | how often | direction | strength | test |
  |---|---|---|---|---|
  | binary | change rate | SPD | DI (`rate_ratio`) | McNemar (asymptotic, no continuity correction) |
  | ordinal | change rate | mean Δ in levels | mean \|Δ\| in levels (`mean_absolute_difference`) | Wilcoxon signed-rank |
  | nominal | change rate | the category whose share changed most (`largest_shift_at`, `largest_shift`) | total variation distance (TVD) | Stuart–Maxwell over the categories that changed, permutation p-value |
  | continuous | change rate | mean Δ | mean \|Δ\| (`mean_absolute_difference`); d_z | paired t-test |

  The change rate is in `metric_value`, the effect size matching the test (SPD, mean Δ, TVD, mean Δ) in
  `effect_value`. Ordinal rows also give the cut-off with the largest change in the share of answers "level ≥ k"
  (`largest_shift_at` = `≥ <level>`, `largest_shift`).

- The tests are fixed in code, one per type: McNemar without continuity correction, its generalization Stuart–Maxwell for categories (only the categories that changed enter, and separate groups of
  categories that exchange cases are summed; its p-value comes from a permutation test with 10,000 random swaps of the
  two versions within the changed cases, because the χ² approximation is too conservative when only a few cases
  change category), Wilcoxon signed-rank for ordered
  levels (unchanged cases left out, as in McNemar) and the paired t-test for the mean difference.
- **The significance level is the only statistical setting** (`statistics.alpha`, default 0.05). The tests,
  Benjamini–Hochberg, the 95% confidence level, 2000 bootstrap resamples and 10,000 permutations are fixed: they are
  the analysis plan, or Monte Carlo precision that does not change what is measured. To re-analyse stored results
  with another α, change `config.statistics.alpha` in `manifest.yaml` and run `analyze`; the new analysis gets its
  own folder (step 8).
- Every metric comes with a 95% confidence interval (`<metric>_ci_lower/upper`; for the main metric and the effect
  size `metric_ci_*` and `effect_ci_*`): the range in which the true value plausibly lies given the number of cases.
  Rates (change rate, selection rates and EOD) use the Wilson score interval; mean Δ,
  mean \|Δ\|, TVD, the largest shifts and DI use a percentile bootstrap that resamples the cases (2000 resamples).
  EOddsD, AOD and d_z have no interval.
- The total variation distance `½ Σ_c |p'_c − p_c|` (`p_c` = share of category `c`) is the share of cases that would
  have to change category to make the two distributions equal; 0 is the null hypothesis of the Stuart–Maxwell test.
  Mean Δ and mean \|Δ\| of ordinal levels count every level step as equal; the change rate
  only uses their order.
  A bootstrap interval is empty for a single case, and for DI when a resample has no yes answer in
  either version (roughly fewer than 10 yes answers).
- `effect_value` keeps the direction (counterfactual − original). `adjusted_p_value` is corrected with
  Benjamini–Hochberg over all tests of the run; the `original->retest` controls are corrected
  separately (column `family`: `primary` or `control`). `statistically_significant` means
  `adjusted_p_value < statistics.alpha`.
- **Group fairness metrics** of binary outputs are computed with
  [Fairlearn](https://fairlearn.org); the two versions of a case are the groups:

  | column | fairness metric (abbreviation) | meaning |
  |---|---|---|
  | `rate_reference`, `rate_counterfactual` | selection rate (`rate ref.`, `rate cf.`) | share of yes answers per version |
  | `effect_value` | statistical parity difference (`SPD`) | `rate_counterfactual − rate_reference` |
  | `rate_ratio` | disparate impact (`DI`) | `rate_counterfactual / rate_reference`; empty if the reference has no yes answer |
  | `equal_opportunity_difference` | equal opportunity difference (`EOD`) | share of the reference yes answers that became no |
  | `equalized_odds_difference` | equalized odds difference (`EOddsD`) | the larger of that and the share of the reference no answers that became yes |
  | `average_odds_difference` | average odds difference (`AOD`) | ½ (share of the reference no answers that became yes − share of the reference yes answers that became no); the signed counterpart of EOddsD |

  Equal opportunity, equalized odds and average odds normally compare the answers with the correct ones. The data has
  no correct answers, so the answer to the reference version is used as the label: they then measure consistency
  with the model's own answer, not correctness. They are descriptive values without a test. Their intervals: Wilson
  for the selection rates and EOD, bootstrap for DI, none for EOddsD and AOD.
- The tests, corrections and resampling come from `scipy` and `statsmodels`.
- The `original->retest` rows (`sensitive_attribute` = `control`) show how much the model changes when **nothing** changes. A
  counterfactual effect should be clearly larger than this instability floor.
- When every pair is identical, the test is reported with p = 1 and the note "all paired differences are zero".
- Binary, ordinal and continuous rows in `tests` also count the cases that changed (`flips`), went down (`n_lower`) and went up (`n_higher`).
- See [Interpreting results](#interpreting-results) before drawing conclusions about bias.

## Configuration reference

All options with their defaults:

```yaml
name: my_experiment                  # results go to <results_dir>/<name>_<timestamp>/
results_dir: results

dataset:
  path: configs/cases.jsonl          # local file or hf://datasets/<owner>/<repo>/<file>
  sheet: null                        # Excel sheet name or index
  text_column: case_template         # column holding the case text
  id_column: null                    # default: case_0001, case_0002, ... (row numbers)
  placeholder: "{{SENSITIVE_ATTRIBUTE}}"  # token in the case text replaced by the sensitive attribute value
  limit: null                        # use only the first n cases
  sample: null                       # use n random cases (seed = generation.seed)

original_value: ""                   # text inserted into the placeholder for the original case
schema: configs/schema.json

prompts:
  system: configs/prompts/system.txt
  user: configs/prompts/user.txt

models:                              # a plain model name from huggingface is enough: - Qwen/Qwen2.5-1.5B-Instruct
  - name: google/medgemma-4b-it
    provider: transformers           # library that provide model
    dtype: null                      # default: bfloat16 if the GPU supports it, else float16; float32 on CPU
    quantization: null               # 4bit | 8bit 
    device: auto                     # auto | cpu
    batch_size: 8                    # prompts generated together; lower it if GPU memory runs out

sensitive_attributes: {}             # required, see step 4

generation:
  temperature: 0.0                   # 0 = greedy decoding (identical output is still not guaranteed)
  max_new_tokens: 512                # longest allowed answer; longer answers are cut off and become INVALID_JSON
  repetitions: 1                     # runs per model × case × condition; > 1 only useful with temperature > 0
  seed: 42                           # run r uses seed + r for every condition of a case

statistics:
  alpha: 0.05                        # significance level of the adjusted p-values

visualization:
  enabled: true
  formats: [pdf, png]                # pdf, svg, png; the report embeds pdf 
  dpi: 300
```

After a run, the resolved config is stored in `manifest.yaml` together with the full prompt and schema text.

## Tests

```bash
python -m pytest
```

The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB) for the end-to-end test; the other tests use
synthetic results.

## Extending

- **Model backend:** any Hugging Face chat model only needs a `models` entry. Another backend (e.g. OpenAI, vLLM):
  subclass `ModelAdapter` in a new file under `src/models/`, implement
  `generate(batch, temperature, max_new_tokens, seed) -> list[str]` (one answer per chat in `batch`) and optionally
  `close()`, add it to `MODELS` in `src/models/base.py` and select it with `provider:` in the model entry.
- **Sensitive attribute:** configuration only (step 4).
- **Dataset source:** files as in step 1. Another file format: add a reader to `READERS` in
  `src/generation/loader.py`.

## Project structure

```
configs/     files of the actual study: llms_bias_bench.yaml, llms_bias_bench_local.yaml, llms_bias_bench/ (schema.json, prompts/), cv_screening
slurm/       FRIDA job script (frida.sbatch) and instructions (README.md)
examples/    template set, not used by the study: example.yaml, cases.jsonl, schema.json, prompts/
src/
  main.py        command line and pipeline stages
  config.py      validation and defaults of the YAML config
  store.py       reading and writing the results folder
  generation/    loader, design, prompts, parsing, runner
  models/        model backends
  evaluation/    dataset, analyze, statistical_tests, metrics
  reporting/     figures, report (data) + report.typ (Typst template of report.pdf)
tests/       pytest suite (small real model + synthetic results)
results/     one folder per experiment
```
