"""Preprocess HotpotQA for tool-agent rollout profiling."""

from __future__ import annotations

import argparse
import os

import datasets


def _answer(example: dict) -> str:
    value = example.get("answer", "")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _question(example: dict) -> str:
    return str(example.get("question", ""))


def _process(split: str):
    def process_fn(example: dict, idx: int) -> dict:
        question = _question(example)
        answer = _answer(example)
        prompt_text = (
            f"{question}\n\n"
            "Use the `retrieve_hotpot_context` tool at least twice before answering. "
            "Then provide a concise final answer inside <answer>...</answer> tags."
        )
        return {
            "data_source": "searchR1_hotpotqa",
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You answer HotpotQA questions. Reason briefly, call "
                        "`retrieve_hotpot_context` to gather evidence, and then answer "
                        "using exactly one <answer>...</answer> block."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            "ability": "qa",
            "reward_model": {"style": "rule", "ground_truth": {"target": answer}},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
                "answer": answer,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "retrieve_hotpot_context": {
                        "create_kwargs": {"ground_truth": {"target": answer}},
                    },
                },
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/hotpotqa_tool_agent")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=1024)
    args = parser.parse_args()

    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path, "distractor")
    else:
        # HF moved the canonical `hotpot_qa` dataset under the `hotpotqa` namespace;
        # the legacy script-based id no longer resolves with datasets>=4.
        dataset = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor")

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"] if "validation" in dataset else dataset["train"]
    if args.train_limit:
        train_dataset = train_dataset.select(range(min(args.train_limit, len(train_dataset))))
    if args.test_limit:
        test_dataset = test_dataset.select(range(min(args.test_limit, len(test_dataset))))

    train_dataset = train_dataset.map(function=_process("train"), with_indices=True)
    test_dataset = test_dataset.map(function=_process("test"), with_indices=True)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))
    print(f"Wrote HotpotQA tool-agent data to {save_dir}")
