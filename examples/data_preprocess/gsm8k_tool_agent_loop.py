# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def _load_raw_split(raw_dir, split):
    """Load a raw GSM8K split (columns: question, answer) from a local directory.

    Supports {split}.parquet / {split}.json / {split}.jsonl. This is the offline
    path for environments without HuggingFace/ModelScope network access.
    """
    for ext, builder in (("parquet", "parquet"), ("jsonl", "json"), ("json", "json")):
        path = os.path.join(raw_dir, f"{split}.{ext}")
        if os.path.exists(path):
            return datasets.load_dataset(builder, data_files=path, split="train")
    raise FileNotFoundError(f"No raw '{split}' file (parquet/jsonl/json) found under {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--raw_dir",
        default=None,
        help="Offline source dir holding raw {train,test}.{parquet,jsonl,json} with columns question/answer.",
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if args.raw_dir is not None:
        # Offline: read the official train/test splits from local raw files.
        train_dataset = _load_raw_split(args.raw_dir, "train")
        test_dataset = _load_raw_split(args.raw_dir, "test")
    else:
        if local_dataset_path is not None:
            dataset = datasets.load_dataset(local_dataset_path, "main")
        else:
            dataset = datasets.load_dataset(data_source, "main")

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer in the format of `#### <answer>`."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": solution},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Leakage guard: train and test questions must be disjoint (no test sample seen in training).
    train_questions = {ex["extra_info"]["question"] for ex in train_dataset}
    test_questions = {ex["extra_info"]["question"] for ex in test_dataset}
    overlap = train_questions & test_questions
    print(f"[gsm8k] train={len(train_dataset)} test={len(test_dataset)} overlap_questions={len(overlap)}")
    assert not overlap, (
        f"Data leakage: {len(overlap)} test question(s) also appear in train. "
        "Refusing to write a contaminated split."
    )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
