"""
Evaluate a judge rubric against a dataset.

Runs the judge on every trace and reports accuracy, precision, recall,
per-task and per-model breakdowns, and a confusion matrix.

Usage:
    from evaluate import evaluate_rubric, call_judge
    metrics = evaluate_rubric(rubric="...", dataset_path="datasets/.../val.json")
"""

import json
import os
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Suppress litellm noise
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
os.environ["LITELLM_LOG"] = "ERROR"

import litellm
litellm.suppress_debug_info = True

from litellm import completion

try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it


# ── Judge LLM call ───────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator of customer service agent conversations.
You will be given a conversation trace and a rubric. Follow the rubric to evaluate
whether the agent correctly adhered to policy.

Return ONLY a JSON object with two fields:
- "verdict": true if the agent was compliant, false if the agent violated policy
- "reasoning": a brief explanation (2-3 sentences) of your judgment"""


def call_judge(
    rubric: str,
    trace: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
) -> dict:
    """
    Call the LLM judge with a rubric and trace.

    Returns: {"verdict": bool, "reasoning": str, "raw": str, "error": str|None}
    """
    prompt = f"""## Rubric
{rubric}

## Conversation Trace
{trace}

Evaluate the agent's policy compliance. Return JSON: {{"verdict": true/false, "reasoning": "..."}}"""

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle ```json ... ``` wrapping)
        text = raw
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        return {
            "verdict": bool(parsed.get("verdict", False)),
            "reasoning": parsed.get("reasoning", ""),
            "raw": raw,
            "error": None,
        }
    except Exception as e:
        return {
            "verdict": False,
            "reasoning": "",
            "raw": str(e),
            "error": str(e),
        }


# ── Metrics computation ─────────────────────────────────────────────────────


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1 from a list of result dicts."""
    tp = fp = tn = fn = 0
    errors = 0

    for r in results:
        if r["error"]:
            errors += 1
            continue
        gt = r["ground_truth"]
        pred = r["verdict"]
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": accuracy,
        "precision_compliant": precision,
        "recall_compliant": recall,
        "f1_compliant": f1,
        "precision_non_compliant": tn / (tn + fn) if (tn + fn) else 0,
        "recall_non_compliant": tn / (tn + fp) if (tn + fp) else 0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total": total,
        "errors": errors,
    }


# ── Main evaluation ─────────────────────────────────────────────────────────


def _eval_one(rubric: str, entry: dict, judge_model: str) -> dict:
    """Evaluate a single trace. Used for parallel execution."""
    judge_result = call_judge(rubric, entry["trace"], model=judge_model)
    return {
        "id": entry["id"],
        "task_id": entry["task_id"],
        "model": entry["model"],
        "trial": entry["trial"],
        "ground_truth": entry["ground_truth"],
        "verdict": judge_result["verdict"],
        "reasoning": judge_result["reasoning"],
        "error": judge_result["error"],
        "correct": judge_result["verdict"] == entry["ground_truth"] if not judge_result["error"] else None,
    }


def evaluate_rubric(
    rubric: str,
    dataset_path: str | Path,
    judge_model: str = "openai/gpt-4o-mini",
    output_dir: str | Path | None = None,
    verbose: bool = True,
    max_workers: int = 20,
) -> dict:
    """
    Run the judge with `rubric` on every trace in the dataset.
    Parallelized with ThreadPoolExecutor.
    """
    dataset_path = Path(dataset_path)

    with open(dataset_path) as f:
        dataset = json.load(f)

    if verbose:
        print(f"Evaluating {len(dataset)} traces with model={judge_model} (workers={max_workers})")

    # Parallel evaluation
    results = [None] * len(dataset)
    correct_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_eval_one, rubric, entry, judge_model): i
            for i, entry in enumerate(dataset)
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Judging")
        for future in pbar:
            i = futures[future]
            result = future.result()
            results[i] = result
            if result["correct"]:
                correct_count += 1
            done = sum(1 for r in results if r is not None)
            pbar.set_postfix(acc=f"{correct_count/done:.0%}")

    # Overall metrics
    overall = compute_metrics(results)

    # Per-task metrics
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)
    per_task = {tid: compute_metrics(rs) for tid, rs in sorted(by_task.items(), key=lambda x: int(x[0]))}

    # Per-model metrics
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    per_model = {m: compute_metrics(rs) for m, rs in sorted(by_model.items())}

    output = {
        "rubric": rubric,
        "judge_model": judge_model,
        "dataset": str(dataset_path),
        "overall": overall,
        "per_task": per_task,
        "per_model": per_model,
        "results": results,
    }

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(output, f, indent=2)

    # Print summary
    if verbose:
        cm = overall["confusion_matrix"]
        print(f"\n{'=' * 60}")
        print(f"  Accuracy:  {overall['accuracy']:.1%}  |  Errors: {overall['errors']}")
        cm = overall["confusion_matrix"]
        print(f"  Compliant:     P={overall['precision_compliant']:.0%} R={overall['recall_compliant']:.0%} F1={overall['f1_compliant']:.0%}")
        nc_p = overall['precision_non_compliant']
        nc_r = overall['recall_non_compliant']
        nc_f1 = 2*nc_p*nc_r/(nc_p+nc_r) if (nc_p+nc_r) else 0
        print(f"  Non-compliant: P={nc_p:.0%} R={nc_r:.0%} F1={nc_f1:.0%}")
        print(f"  Confusion: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
        says_true = cm['tp'] + cm['fp']
        if overall['total']:
            print(f"  Bias: says compliant {says_true}/{overall['total']} ({says_true/overall['total']:.0%})")
        elif overall['errors']:
            first_error = next((r.get('error') for r in results if r.get('error')), None)
            print("  Bias: unavailable because all judge calls errored")
            if first_error:
                print(f"  First error: {first_error}")
        print(f"{'=' * 60}")

    return output


if __name__ == "__main__":
    import sys

    rubric = "Determine whether the customer service agent correctly followed all applicable policies during this conversation."
    dataset = sys.argv[1] if len(sys.argv) > 1 else "workshop/data/airline_policy_v0/val.json"
    evaluate_rubric(rubric, dataset)
