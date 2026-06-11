"""Preprocess HotpotQA for a Search-R1 style ReAct tool-agent rollout.

Unlike `hotpotqa_tool_agent_loop.py` (which pairs with the synthetic
fixed-context tool), this builds a Search-R1 / ReAct system prompt intended for
use with the REAL BM25 retriever tool (`hotpot_online_search_tool.py` ->
`retrieve_hotpot_context`), which returns actual Wikipedia passages wrapped in
<information>...</information>.
"""

from __future__ import annotations

import argparse
import os

import datasets

SYSTEM_PROMPT = (
    "You are a research assistant that answers multi-hop questions by searching a "
    "Wikipedia knowledge base.\n\n"
    "You have one tool, `retrieve_hotpot_context`, which takes a focused search "
    "`query` and returns the most relevant Wikipedia passages wrapped in "
    "<information> and </information>.\n\n"
    "Follow this loop on every step:\n"
    "1. Reason about what you still need to know inside <think> and </think>.\n"
    "2. If you lack evidence, call `retrieve_hotpot_context` with a single, focused "
    "query (one entity or relation at a time). You may search multiple times, "
    "refining the query using what previous results told you.\n"
    "3. Read the returned <information> before deciding the next step.\n"
    "4. Once you have enough evidence, stop searching and give the final answer "
    "inside <answer> and </answer>, as short as possible (a name, entity, number, "
    'or yes/no), e.g. <answer>Beijing</answer>. Do not add explanations inside the '
    "answer tags.\n\n"
    "Always think before you search or answer."
)


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
        user_text = (
            f"Question: {question}\n\n"
            "Search the knowledge base for the evidence you need, then give the "
            "final short answer inside <answer>...</answer>."
        )
        return {
            "data_source": "searchR1_hotpotqa",
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
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
    parser.add_argument("--local_save_dir", default="~/data/hotpotqa_search_r1_react")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=1024)
    args = parser.parse_args()

    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path, "distractor")
    else:
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
    print(f"Wrote HotpotQA Search-R1 ReAct data to {save_dir}")
