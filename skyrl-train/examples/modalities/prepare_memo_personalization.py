"""
Prepare the MeMo personalization dataset in SkyRL format with a memory modality payload.

This script is a reference only (no environment is created). It maps each user config to a user_id,
injects a memory placeholder token into the system prompt, and stores the user_id as the modality payload.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import datasets


try:
    from memo.dataset.prompts import PERSONA_PROMPT  # type: ignore
except Exception:
    PERSONA_PROMPT = (
        "You are a personalized AI assistant. Answer the question about the user based on your understanding of the user."
    )


def build_sample(
    example: Dict[str, Any],
    idx: int,
    *,
    user_id: int,
    modality_id: str,
    placeholder_token: str,
    question_key: str,
    answer_key: str,
    env_class: str,
) -> Dict[str, Any]:
    question = str(example[question_key]).strip()
    answer = str(example[answer_key]).strip()

    system_prompt = f"{PERSONA_PROMPT}\nMemory{placeholder_token}Memory"

    return {
        "data_source": "MemoryAsModality/Personalization",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "env_class": env_class,
        "reward_spec": {
            "method": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "index": idx,
            "user_id": user_id,
            "answer": answer,
        },
        "modalities": {
            modality_id: {"user_id": user_id},
        },
    }


def load_user_split(
    dataset_name: str,
    config: str,
    split: str,
) -> datasets.Dataset:
    dataset = datasets.load_dataset(dataset_name, name=config)
    if split in dataset:
        return dataset[split]
    # fallback to train split if missing
    return dataset["train"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="MemoryAsModality/Personalization")
    parser.add_argument("--dataset_configs", nargs="+", required=True)
    parser.add_argument("--question_key", default="question_with_choices")
    parser.add_argument("--answer_key", default="correct_choice")
    parser.add_argument("--placeholder_token", default="<|memory|>")
    parser.add_argument("--modality_id", default="memo_memory")
    parser.add_argument("--env_class", default="personalization")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_sets: List[datasets.Dataset] = []
    val_sets: List[datasets.Dataset] = []

    for user_id, config in enumerate(args.dataset_configs):
        train_split = load_user_split(args.dataset_name, config, "train")
        val_split = load_user_split(args.dataset_name, config, "validation")

        train_sets.append(
            train_split.map(
                lambda ex, idx, uid=user_id: build_sample(
                    ex,
                    idx,
                    user_id=uid,
                    modality_id=args.modality_id,
                    placeholder_token=args.placeholder_token,
                    question_key=args.question_key,
                    answer_key=args.answer_key,
                    env_class=args.env_class,
                ),
                with_indices=True,
                desc=f"Formatting train split for {config}",
            )
        )

        val_sets.append(
            val_split.map(
                lambda ex, idx, uid=user_id: build_sample(
                    ex,
                    idx,
                    user_id=uid,
                    modality_id=args.modality_id,
                    placeholder_token=args.placeholder_token,
                    question_key=args.question_key,
                    answer_key=args.answer_key,
                    env_class=args.env_class,
                ),
                with_indices=True,
                desc=f"Formatting validation split for {config}",
            )
        )

    train_dataset = datasets.concatenate_datasets(train_sets)
    val_dataset = datasets.concatenate_datasets(val_sets)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Wrote train split to {train_path}")
    print(f"Wrote validation split to {val_path}")


if __name__ == "__main__":
    main()
