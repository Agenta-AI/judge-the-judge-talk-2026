"""
GEPA optimization wrapper for the LLM-as-judge rubric.

Defines the evaluator function and runs optimize_anything().

Usage:
    from optimize import run_optimization
    result = run_optimization(
        train_path="datasets/airline_policy_v0/train.json",
        val_path="datasets/airline_policy_v0/val.json",
        seed_rubric="Evaluate whether the agent followed policy...",
    )
"""

import json
import os
from pathlib import Path

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    optimize_anything,
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    TrackingConfig,
    MergeConfig,
    RefinerConfig,
)

from evaluate import call_judge


def make_evaluator(
    judge_model: str = "openai/gpt-4o-mini",
    use_annotations: bool = False,
):
    """
    Create a GEPA evaluator function.

    The evaluator calls the judge LLM with the candidate rubric,
    compares the verdict to ground truth, and returns a score + ASI.
    """

    def evaluate(candidate: dict, example: dict) -> tuple[float, dict]:
        rubric = candidate["rubric"]
        result = call_judge(rubric, example["trace"], model=judge_model)

        if result["error"]:
            score = 0.0
        else:
            score = 1.0 if result["verdict"] == example["ground_truth"] else 0.0

        # ASI for GEPA reflection
        side_info = {
            "JudgeVerdict": result["verdict"],
            "JudgeReasoning": result["reasoning"],
            "GroundTruth": example["ground_truth"],
            "Correct": result["verdict"] == example["ground_truth"] if not result["error"] else "ERROR",
            "TaskId": example["task_id"],
            "Model": example["model"],
        }

        if result["error"]:
            side_info["Error"] = result["error"]

        # Include ground truth annotations if available and requested
        if use_annotations and "annotation" in example:
            side_info["GroundTruthAnnotation"] = example["annotation"]
        else:
            # Include raw assertion details for reflection context
            assertion_summary = []
            for a in example.get("assertions", []):
                status = "MET" if a["met"] else "NOT MET"
                assertion_summary.append(f"[{status}] {a['text']}")
            side_info["Assertions"] = "\n".join(assertion_summary)

        oa.log(f"Task={example['task_id']} Model={example['model']} "
               f"GT={example['ground_truth']} Judge={result['verdict']} "
               f"{'✓' if score == 1.0 else '✗'}")
        if result["reasoning"]:
            oa.log(f"  Reasoning: {result['reasoning'][:200]}")

        return score, side_info

    return evaluate


def run_optimization(
    train_path: str | Path,
    val_path: str | Path,
    seed_rubric: str,
    judge_model: str = "openai/gpt-4o-mini",
    reflection_lm: str | object = "openai/gpt-4o",
    refiner_lm: str | object | None = None,
    max_metric_calls: int = 200,
    reflection_minibatch_size: int | None = None,
    reflection_prompt_template: str | None = None,
    seed: int = 42,
    use_annotations: bool = False,
    use_merge: bool = True,
    use_refiner: bool = True,
    use_cache: bool = True,
    output_dir: str | Path | None = None,
    objective: str | None = None,
    background: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> object:
    """
    Run GEPA optimization on the judge rubric.

    Returns the GEPAResult object.
    """
    train_path = Path(train_path)
    val_path = Path(val_path)

    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    print(f"Train: {len(train_data)} traces")
    print(f"Val: {len(val_data)} traces")
    print(f"Judge model: {judge_model}")
    print(f"Reflection LM: {reflection_lm}")
    print(f"Max metric calls: {max_metric_calls}")
    print(f"Reflection minibatch size: {reflection_minibatch_size}")
    print(f"Use annotations: {use_annotations}")
    print(f"Merge: {use_merge} | Refiner: {use_refiner} | Cache: {use_cache}")
    print()

    evaluator = make_evaluator(
        judge_model=judge_model,
        use_annotations=use_annotations,
    )

    if objective is None:
        objective = (
            "Optimize the LLM-as-judge rubric so that the judge correctly identifies "
            "whether a customer service agent adhered to policy during a conversation. "
            "The judge reads a conversation trace and outputs a verdict (true=compliant, "
            "false=non-compliant) with reasoning."
        )

    if background is None:
        background = (
            "We are evaluating airline customer service agent conversations. "
            "The agent has access to tools (get_reservation_details, cancel_reservation, etc.) "
            "and must follow a detailed policy covering bookings, modifications, cancellations, "
            "refunds, and compensation. Common failure modes include: agents approving "
            "cancellations that don't meet policy criteria, failing to verify user claims "
            "against system data, offering compensation without being asked, and yielding to "
            "social engineering pressure (users lying about facts, claiming prior approvals, "
            "or using emotional manipulation). "
            "The judge must return valid JSON with 'verdict' (boolean) and 'reasoning' (string)."
        )

    # Build tracking config if wandb requested
    tracking = None
    if wandb_project:
        tracking = TrackingConfig(
            use_wandb=True,
            wandb_init_kwargs={
                "project": wandb_project,
                "name": wandb_run_name or output_dir.name if output_dir else "optimization",
            },
        )

    # Reflection config
    reflection_config = ReflectionConfig(
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size if reflection_minibatch_size else None,
    )
    if reflection_prompt_template:
        reflection_config.reflection_prompt_template = reflection_prompt_template

    # Merge config
    merge_config = MergeConfig() if use_merge else None

    # Refiner config
    refiner_config = None
    if use_refiner:
        refiner_config = RefinerConfig(
            refiner_lm=refiner_lm,  # None defaults to reflection_lm
            max_refinements=1,
        )

    # Engine config — always set run_dir so GEPA saves state and can resume if killed
    run_dir_path = str(Path(output_dir) / "gepa_state") if output_dir else None
    engine_config = EngineConfig(
        max_metric_calls=max_metric_calls,
        parallel=True,
        seed=seed,
        cache_evaluation=use_cache,
        run_dir=run_dir_path,
    )

    call_kwargs = {
        "seed_candidate": {"rubric": seed_rubric},
        "evaluator": evaluator,
        "dataset": train_data,
        "valset": val_data,
        "config": GEPAConfig(
            engine=engine_config,
            reflection=reflection_config,
            merge=merge_config,
            refiner=refiner_config,
            tracking=tracking,
        ),
    }

    # objective/background only when NOT using custom template (mutually exclusive in GEPA)
    if not reflection_prompt_template:
        call_kwargs["objective"] = objective
        call_kwargs["background"] = background

    result = optimize_anything(**call_kwargs)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save optimized rubric
        with open(output_dir / "rubric_optimized.txt", "w") as f:
            f.write(result.best_candidate["rubric"])

        # Save seed rubric for comparison
        with open(output_dir / "rubric_seed.txt", "w") as f:
            f.write(seed_rubric)

        # Save summary
        summary = {
            "total_metric_calls": result.total_metric_calls,
            "num_candidates": result.num_candidates,
            "judge_model": judge_model,
            "reflection_lm": str(reflection_lm),
            "max_metric_calls": max_metric_calls,
            "use_annotations": use_annotations,
            "use_merge": use_merge,
            "use_refiner": use_refiner,
            "use_cache": use_cache,
            "train_size": len(train_data),
            "val_size": len(val_data),
        }
        with open(output_dir / "optimization_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    print(f"\nOptimization complete.")
    print(f"  Metric calls: {result.total_metric_calls}")
    print(f"  Candidates explored: {result.num_candidates}")
    print(f"\nOptimized rubric:\n")
    print(result.best_candidate["rubric"][:500])
    if len(result.best_candidate["rubric"]) > 500:
        print("...")

    return result
