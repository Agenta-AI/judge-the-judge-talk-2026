# Judge the Judge: Workshop Assets

Public assets for the workshop **"Judge the Judge: Building LLM Evaluators That Actually Work."**

This repo is centered around the notebook in `workshop/01_build_judge_with_gepa.ipynb`.

It includes:
- the workshop notebook
- the processed airline policy dataset
- the core scripts used by the notebook
- curated experiment outputs and rubrics
- reproducible experiment entrypoints

## Resources

- GEPA repo: `https://github.com/gepa-ai/gepa`
- GEPA paper: `https://arxiv.org/abs/2507.19457`
- Tau-bench repo: `https://github.com/sierra-research/tau-bench`

## Quickstart

### 1. Install dependencies

This repo uses `uv`.

```bash
uv sync
```

### 2. Start Jupyter

```bash
uv run jupyter lab
```

Then open:

```text
workshop/01_build_judge_with_gepa.ipynb
```

## Recommended Path

The notebook is the main workshop experience.

By default, it uses **checked-in results** for the baseline and optimized judge so it can run without API credentials.

That means you can:
- open the notebook
- inspect the dataset
- see the baseline failure mode
- load the optimized rubric
- compare seed vs optimized results
- inspect flipped examples and remaining failures

without rerunning expensive model calls.

## What The Notebook Covers

The notebook walks through:

1. problem framing: why naive LLM judges fail
2. loading and inspecting the dataset
3. baseline judge behavior
4. GEPA optimization setup
5. loading the optimized rubric
6. before/after comparison
7. failure analysis
8. takeaways and practical recipe

## Repo Structure

```text
.
├── workshop/
│   ├── 01_build_judge_with_gepa.ipynb
│   ├── core/
│   │   ├── annotate.py
│   │   ├── evaluate.py
│   │   ├── extract.py
│   │   ├── optimize.py
│   │   └── split.py
│   ├── data/
│   │   └── airline_policy_v0/
│   └── results/
│       ├── baseline/
│       ├── best/
│       └── slide_metrics/
├── experiments/
├── slides/
└── docs/
```

## Core Folders

### `workshop/`

The main workshop assets.

### `workshop/core/`

Human-readable helper scripts used by the notebook and experiments:
- `extract.py`: build the dataset from raw Tau2-bench traces
- `split.py`: split by task to avoid leakage
- `annotate.py`: generate trace-level annotations
- `evaluate.py`: run a judge rubric on a dataset
- `optimize.py`: run GEPA prompt optimization for the judge rubric

### `workshop/data/airline_policy_v0/`

Processed dataset used in the workshop.

Files included:
- `train.json`
- `train_annotated.json`
- `val.json`
- `val_annotated.json`
- `full.json`
- `full_annotated.json`
- `policy.md`

### `workshop/results/`

Curated outputs used by the notebook.

Important files:
- `baseline/eval_results.json`
- `best/rubric_seed.txt`
- `best/rubric_optimized.txt`
- `best/eval_val.json`
- `best/optimization_summary.json`

### `experiments/`

Reproducible experiment scripts.

These are not required for the main notebook flow. They exist for people who want to rerun the ablations or the best experiment from the command line.

## Running Experiments

If you want to rerun live model calls, copy `.env.example` to `.env` and set the required API keys.

Then you can run, for example:

```bash
uv run python experiments/exp_001_baseline.py
```

Or the best experiment:

```bash
uv run python experiments/exp_007_grok_gemini.py
```

Live experiment runs write outputs to `runs/` so they do not overwrite the curated checked-in results used by the notebook.

## Notes On Credentials

The notebook is designed to work without API credentials because it loads saved results by default.

You only need credentials if you want to rerun:
- judge evaluations
- annotation generation
- GEPA optimization experiments

Depending on model choice, you may need keys and credits for providers such as OpenAI or OpenRouter.

## Main Result

The core workshop story is the shift from a naive judge that mostly rubber-stamps traces as compliant to an optimized judge that catches many more real violations.

Checked-in validation results show:
- baseline accuracy: `65.2%`
- optimized accuracy: `69.6%`
- baseline non-compliant recall: `14.0%`
- optimized non-compliant recall: `55.8%`

That non-compliant recall jump is the main point of the workshop.

## Slides

The slide deck PDF lives in `slides/`.

## License

MIT.
