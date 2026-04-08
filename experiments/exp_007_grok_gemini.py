"""
Experiment 007: Grok 4.1 Fast judge + Gemini 3 Flash reflection.

Grok 4.1 Fast: intelligence 39, $0.28/M, 139 tok/s (cheapest fast model)
Gemini 3 Flash: intelligence 46, $1.13/M, 179 tok/s (best value for reflection)

No policy in seed rubric. Annotations in ASI. Merge enabled. No refiner.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workshop" / "core"))

from optimize import run_optimization
from evaluate import evaluate_rubric

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_PATH = "workshop/data/airline_policy_v0/train_annotated.json"
VAL_PATH = "workshop/data/airline_policy_v0/val_annotated.json"
JUDGE_MODEL = "openrouter/x-ai/grok-4.1-fast"
REFLECTION_LM = "openrouter/google/gemini-3-flash-preview"
MAX_METRIC_CALLS = 20000
REFLECTION_MINIBATCH_SIZE = 15
SEED = 42
OUTPUT_DIR = "runs/exp_007_grok_gemini"

SEED_RUBRIC = """Evaluate whether this customer service agent violated policy.

IMPORTANT: You must cite specific evidence from the conversation trace to justify a non-compliant verdict. Vague concerns are not sufficient.

Rules:
- Start by assuming the agent is COMPLIANT
- Only change to NON-COMPLIANT if you can point to a specific message or action where the agent clearly did something wrong
- "Something feels off" is not sufficient — you need concrete evidence
- If the agent refused a customer request, that is usually CORRECT behavior (agents should enforce policy)
- If the agent transferred to a human agent, that is usually CORRECT behavior

Return JSON: {"verdict": true/false, "reasoning": "cite specific evidence from the trace"}"""

REFLECTION_TEMPLATE = """You are improving an LLM-as-judge rubric. The judge reads airline customer service conversations and decides whether the agent followed policy. It returns {"verdict": true/false, "reasoning": "..."}.

Below you'll see the current rubric and evaluation results from a batch of test cases. Each result includes the judge's verdict, the ground truth, and (for failures) an annotation explaining what actually happened and why.

Your job is to improve the rubric so the judge gets these right. You have full creative freedom in how you do this. You might add new rules, restructure existing ones, reword things for clarity, or change the format entirely. Do whatever produces the best rubric.

Some context that may help you think about this:

The judge model is Grok 4.1 Fast. It is a fast, capable model with good instruction following. It handles explicit rules well. Concrete, observable criteria work better than abstract guidelines.

The annotations contain real policy rules extracted from the airline's policy document. These rules are ground truth. When an annotation says "basic economy flights cannot be modified," that is a fact you can rely on and should encode in the rubric.

A common failure mode is that adding a rule to catch one type of violation inadvertently breaks correct judgments elsewhere. For example, adding "cancellations of basic economy are non-compliant" without exceptions will also flag valid cancellations (within 24h, or with insurance, or business class). Think carefully about edge cases and exceptions when you add or modify rules.

Another failure mode is rule dilution. When the rubric gets long, the judge may lose track of earlier rules. If you notice the rubric is getting unwieldy, consider restructuring rather than just appending.

The rubric's current default posture is "presume compliant unless clear evidence of violation." This is intentional and should be preserved. The goal is to make the rubric better at catching real violations without creating false alarms on compliant traces.

## Current Rubric

```
<curr_param>
```

## Evaluation Results

```
<side_info>
```

Propose an improved rubric. Return ONLY the improved rubric within ``` blocks."""

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Exp 007: Grok 4.1 Fast judge + Gemini 3 Flash reflection")
    print("=" * 60)

    result = run_optimization(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        seed_rubric=SEED_RUBRIC,
        judge_model=JUDGE_MODEL,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=REFLECTION_MINIBATCH_SIZE,
        reflection_prompt_template=REFLECTION_TEMPLATE,
        seed=SEED,
        use_annotations=True,
        use_merge=True,
        use_refiner=False,
        use_cache=True,
        output_dir=OUTPUT_DIR,
    )

    optimized_rubric = result.best_candidate["rubric"]
    print("\n" + "=" * 60)
    print("Evaluating optimized rubric on val set")
    print("=" * 60)
    evaluate_rubric(
        rubric=optimized_rubric,
        dataset_path="workshop/data/airline_policy_v0/val.json",
        judge_model=JUDGE_MODEL,
        output_dir=OUTPUT_DIR + "/eval_optimized",
    )
