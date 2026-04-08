"""
Extract policy-adherence evaluation data from tau2-bench simulation results.

Produces a dataset of (trace, ground_truth) pairs where:
- Each trace is a cleaned conversation between agent and user
- Ground truth is whether the agent adhered to policy (aggregated from per-assertion labels)
- Only policy-relevant assertions are considered (filtered by tag)

Usage:
    from extract import extract_dataset
    extract_dataset()  # uses defaults
    extract_dataset(output_dir="datasets/my_variant", include_tags={"social_engineering_resistance"})
"""

import json
import glob
import os
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

DEFAULT_RESULTS_DIR = BASE_DIR / "tau2-bench" / "data" / "tau2" / "results" / "final"
DEFAULT_POLICY_PATH = BASE_DIR / "tau2-bench" / "data" / "tau2" / "domains" / "airline" / "policy.md"
DEFAULT_LABELED_PATH = BASE_DIR / "analysis" / "tau2_eval_labeled.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "airline_policy_v0"

DEFAULT_DOMAIN = "airline"
DEFAULT_INCLUDE_TAGS = {
    "social_engineering_resistance",
    "policy_compliance",
    "refusal_behavior",
    "information_verification",
    "proactive_behavior_restraint",
    "data_cross_reference",
}


# ── Trace cleaning ───────────────────────────────────────────────────────────


def clean_trace(messages: list[dict]) -> str:
    """
    Convert raw tau2 messages into a clean, readable conversation string.

    Strips metadata (timestamp, cost, usage, raw_data).
    Formats as:
        Assistant: ...
        Human: ...
        [Tool Call: func_name(arg1="val1", arg2="val2")]
        [Tool Result: ...]
    """
    parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", None)

        if role == "assistant":
            if content.strip():
                parts.append(f"Assistant: {content.strip()}")
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("arguments", {})
                    args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                        for k, v in args.items())
                    parts.append(f"[Tool Call: {name}({args_str})]")

        elif role == "user":
            parts.append(f"Human: {content.strip()}")

        elif role == "tool":
            # Truncate very long tool results (some have 10K+ chars)
            result_str = content.strip()
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "\n... (truncated)"
            parts.append(f"[Tool Result: {result_str}]")

    return "\n\n---\n\n".join(parts)


# ── Tag index ────────────────────────────────────────────────────────────────


def build_tag_index(labeled_path: str | Path, domain: str) -> dict:
    """
    Build a lookup: (task_id, assertion_idx) → list of tags.

    Uses the labeled assertions file where IDs are like 'airline_2_nl_0'.
    """
    with open(labeled_path) as f:
        labeled = json.load(f)

    index = {}
    for item in labeled["labeled_nl_assertions"]:
        if item["domain"] != domain:
            continue
        # Parse index from ID: 'airline_2_nl_0' → task_id='2', idx=0
        parts = item["id"].split("_")
        # Format: {domain}_{task_id}_nl_{idx}
        task_id = parts[1]
        assertion_idx = int(parts[-1])
        index[(task_id, assertion_idx)] = item["tags"]

    return index


# ── Main extraction ──────────────────────────────────────────────────────────


def extract_dataset(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    labeled_path: str | Path = DEFAULT_LABELED_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    domain: str = DEFAULT_DOMAIN,
    include_tags: set[str] = DEFAULT_INCLUDE_TAGS,
) -> Path:
    """
    Extract policy-adherence evaluation data from tau2-bench results.

    Returns the path to the output file.
    """
    results_dir = Path(results_dir)
    policy_path = Path(policy_path)
    labeled_path = Path(labeled_path)
    output_dir = Path(output_dir)

    # Load policy
    policy_text = policy_path.read_text()

    # Build tag index
    tag_index = build_tag_index(labeled_path, domain)

    # Find simulation result files
    result_files = sorted(glob.glob(str(results_dir / f"*{domain}*")))
    if not result_files:
        raise FileNotFoundError(f"No {domain} result files found in {results_dir}")

    print(f"Found {len(result_files)} result files")
    print(f"Policy: {len(policy_text)} chars")
    print(f"Tag index: {len(tag_index)} (task, assertion) pairs")
    print(f"Include tags: {include_tags}")
    print()

    # Extract traces
    dataset = []
    stats = {"total_sims": 0, "skipped_no_policy_assertions": 0,
             "included": 0, "compliant": 0, "non_compliant": 0}

    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        model = data["info"].get("agent_info", {}).get("llm", "unknown")
        print(f"Processing {model} ({len(data['simulations'])} simulations)...")

        for sim in data["simulations"]:
            stats["total_sims"] += 1
            task_id = str(sim["task_id"])
            trial = sim["trial"]
            nl_assertions = sim["reward_info"].get("nl_assertions", [])

            # Filter to policy-relevant assertions
            policy_assertions = []
            for idx, nla in enumerate(nl_assertions):
                tags = tag_index.get((task_id, idx), [])
                if include_tags.intersection(tags):
                    policy_assertions.append({
                        "text": nla["nl_assertion"],
                        "met": nla["met"],
                        "justification": nla.get("justification", ""),
                        "tags": tags,
                        "index": idx,
                    })

            # Skip traces with no policy-relevant assertions
            if not policy_assertions:
                stats["skipped_no_policy_assertions"] += 1
                continue

            # Aggregate ground truth: false if ANY policy assertion fails
            ground_truth = all(a["met"] for a in policy_assertions)

            # Clean trace
            trace = clean_trace(sim["messages"])

            entry = {
                "id": f"{model}_task{task_id}_trial{trial}",
                "task_id": task_id,
                "model": model,
                "trial": trial,
                "trace": trace,
                "policy": policy_text,
                "ground_truth": ground_truth,
                "assertions": policy_assertions,
                "num_messages": len(sim["messages"]),
            }
            dataset.append(entry)
            stats["included"] += 1
            if ground_truth:
                stats["compliant"] += 1
            else:
                stats["non_compliant"] += 1

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Report
    print()
    print("=" * 60)
    print("Extraction complete")
    print("=" * 60)
    print(f"Total simulations scanned:  {stats['total_sims']}")
    print(f"Skipped (no policy asserts): {stats['skipped_no_policy_assertions']}")
    print(f"Included in dataset:         {stats['included']}")
    print(f"  Compliant (true):          {stats['compliant']} ({stats['compliant']/stats['included']*100:.1f}%)")
    print(f"  Non-compliant (false):     {stats['non_compliant']} ({stats['non_compliant']/stats['included']*100:.1f}%)")
    print(f"Unique tasks:                {len(set(e['task_id'] for e in dataset))}")
    print(f"Unique models:               {len(set(e['model'] for e in dataset))}")
    print(f"Output: {output_path}")

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_dataset()
