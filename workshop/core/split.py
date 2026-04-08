"""
Split a dataset into train/val at the task level.

Ensures no data leakage: all traces for a given task go entirely into
train or val. Stratifies so that val covers diverse assertion tag types.

Usage:
    from split import split_dataset
    split_dataset("workshop/data/airline_policy_v0/full.json")
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def split_dataset(
    dataset_path: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Split dataset by task_id. Returns (train_path, val_path).
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir else dataset_path.parent

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Group by task_id
    tasks = defaultdict(list)
    for entry in dataset:
        tasks[entry["task_id"]].append(entry)

    task_ids = sorted(tasks.keys(), key=int)
    num_val = max(1, round(len(task_ids) * val_ratio))

    # Collect all tags per task for stratification info
    task_tags = {}
    task_balance = {}
    for tid in task_ids:
        all_tags = set()
        compliant = 0
        total = 0
        for entry in tasks[tid]:
            for a in entry["assertions"]:
                all_tags.update(a["tags"])
            if entry["ground_truth"]:
                compliant += 1
            total += 1
        task_tags[tid] = all_tags
        task_balance[tid] = (compliant, total)

    # Shuffle and pick val tasks
    rng = random.Random(seed)
    shuffled = list(task_ids)
    rng.shuffle(shuffled)

    val_task_ids = set(shuffled[:num_val])
    train_task_ids = set(shuffled[num_val:])

    # Build splits
    train_data = [e for e in dataset if e["task_id"] in train_task_ids]
    val_data = [e for e in dataset if e["task_id"] in val_task_ids]

    # Write
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

    # Report
    def report_split(name, data, tids):
        compliant = sum(1 for e in data if e["ground_truth"])
        non_compliant = len(data) - compliant
        tags = set()
        for tid in tids:
            tags.update(task_tags[tid])
        pct = compliant / len(data) * 100 if data else 0
        print(f"  {name}:")
        print(f"    Tasks: {len(tids)} {sorted(tids, key=int)}")
        print(f"    Traces: {len(data)}")
        print(f"    Compliant: {compliant} ({pct:.1f}%)")
        print(f"    Non-compliant: {non_compliant} ({100-pct:.1f}%)")
        print(f"    Tags covered: {sorted(tags)}")

    print("=" * 60)
    print("Split complete")
    print("=" * 60)
    report_split("Train", train_data, train_task_ids)
    print()
    report_split("Val", val_data, val_task_ids)
    print(f"\n  Output: {output_dir}")

    return train_path, val_path


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "workshop/data/airline_policy_v0/full.json"
    split_dataset(path)
