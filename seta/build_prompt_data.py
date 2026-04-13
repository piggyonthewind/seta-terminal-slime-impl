"""
Convert SETA task.yaml files into a SLIME-compatible prompt JSONL.

Each line:
  {"text": "<instruction>", "metadata": {"task_path": "<abs_path_to_task_dir>"}}

Usage:
  python seta/build_prompt_data.py \
      --dataset-dir data/seta-env/Dataset \
      --output data/seta_prompts.jsonl
"""

import argparse
import json
import os
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/seta-env/Dataset")
    parser.add_argument("--output", default="data/seta_prompts.jsonl")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    skipped = 0
    written = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for task_id in sorted(os.listdir(dataset_dir), key=lambda x: int(x) if x.isdigit() else x):
            task_path = os.path.join(dataset_dir, task_id)
            yaml_path = os.path.join(task_path, "task.yaml")

            if not os.path.isfile(yaml_path):
                skipped += 1
                continue

            with open(yaml_path, encoding="utf-8") as f:
                task = yaml.safe_load(f)

            instruction = task.get("instruction", "").strip()
            if not instruction:
                skipped += 1
                continue

            record = {
                "text": instruction,
                "metadata": {"task_path": task_path.replace("\\", "/")},
            }
            out.write(json.dumps(record) + "\n")
            written += 1

    print(f"Written {written} tasks to {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
