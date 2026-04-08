"""
Generate trace-level annotations for the dataset.

For each trace, calls an LLM to produce a human-readable explanation of WHY
the trace is compliant or non-compliant, referencing specific policy sections
and conversation moments. These annotations enrich the GEPA training signal.

Usage:
    from annotate import annotate_dataset
    annotate_dataset("workshop/data/airline_policy_v0/full.json")
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from litellm import completion


ANNOTATION_PROMPT = """You are an expert airline customer service auditor. You are given:
1. The airline's agent policy
2. A conversation trace between an agent and a customer
3. Per-assertion evaluation results (each stating whether the agent met a specific behavioral expectation)

Your task: Write a concise trace-level annotation explaining WHY this trace is {verdict_word}.

Guidelines:
- Reference specific policy sections that are relevant
- Point to specific moments in the conversation where the agent acted correctly or incorrectly
- If non-compliant: explain exactly what the agent did wrong and what it should have done
- If compliant: explain what the agent did right, especially if the situation was tricky (e.g., user pressure, lies)
- Keep it to 2-4 sentences
- Be specific, not generic

## Policy
{policy}

## Conversation Trace
{trace}

## Assertion Results
{assertions_text}

## Overall Verdict: {verdict}

Write your annotation:"""


def generate_annotation(
    entry: dict,
    model: str = "openai/gpt-4o-mini",
) -> str:
    """Generate a trace-level annotation for a single entry."""
    verdict = "COMPLIANT" if entry["ground_truth"] else "NON-COMPLIANT"
    verdict_word = "compliant (the agent followed policy)" if entry["ground_truth"] else "non-compliant (the agent violated policy)"

    assertions_text = ""
    for a in entry["assertions"]:
        status = "MET" if a["met"] else "NOT MET"
        assertions_text += f"- [{status}] {a['text']}\n"
        if a.get("justification"):
            assertions_text += f"  Reason: {a['justification']}\n"

    # Truncate trace if very long (keep under ~6K tokens for the annotation call)
    trace = entry["trace"]
    if len(trace) > 12000:
        trace = trace[:12000] + "\n\n... (truncated)"

    prompt = ANNOTATION_PROMPT.format(
        policy=entry["policy"],
        trace=trace,
        assertions_text=assertions_text,
        verdict=verdict,
        verdict_word=verdict_word,
    )

    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def annotate_dataset(
    dataset_path: str | Path,
    model: str = "openai/gpt-4o-mini",
    output_suffix: str = "_annotated",
    batch_size: int = 10,
) -> Path:
    """
    Annotate every entry in the dataset with a trace-level explanation.

    Writes a new file with the same name + suffix (e.g., full_annotated.json).
    Supports resuming: if the output file exists, skips already-annotated entries.
    """
    dataset_path = Path(dataset_path)
    output_path = dataset_path.with_stem(dataset_path.stem + output_suffix)

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Resume support: load existing annotations if any
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            for entry in json.load(f):
                if "annotation" in entry:
                    existing[entry["id"]] = entry["annotation"]
        print(f"Resuming: {len(existing)} entries already annotated")

    total = len(dataset)
    annotated_count = 0
    error_count = 0

    for i, entry in enumerate(dataset):
        # Skip if already annotated
        if entry["id"] in existing:
            entry["annotation"] = existing[entry["id"]]
            annotated_count += 1
            continue

        try:
            annotation = generate_annotation(entry, model=model)
            entry["annotation"] = annotation
            annotated_count += 1
        except Exception as e:
            print(f"  Error on {entry['id']}: {e}")
            entry["annotation"] = f"ERROR: {e}"
            error_count += 1

        # Progress + periodic save
        if (i + 1) % batch_size == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] annotated={annotated_count} errors={error_count}")
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)

    print(f"\nDone. {annotated_count} annotated, {error_count} errors.")
    print(f"Output: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "workshop/data/airline_policy_v0/full.json"
    model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-4o-mini"
    annotate_dataset(path, model=model)
