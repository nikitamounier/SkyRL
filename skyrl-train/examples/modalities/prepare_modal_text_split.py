"""
Create a GSM8K variant with a synthetic "text modality" for the first half of the samples.
Each modified prompt appends a placeholder token, and the modality payload stores token ids for
the original question so the modality pipeline can re-embed them.
"""

import argparse
import os
from typing import Dict, Any

import datasets
from transformers import AutoTokenizer


def build_sample(example: Dict[str, Any], idx: int, split: str, tokenizer, placeholder: str, add_modality: bool):
    question_raw = example.pop("question")
    answer_raw = example.pop("answer")

    content = question_raw + " Let's think step by step and output the final answer after \"####\"."
    content_with_placeholder = f"{content} {placeholder}" if add_modality else content

    data = {
        "data_source": "openai/gsm8k",
        "prompt": [
            {"role": "user", "content": content_with_placeholder}
        ],
        "env_class": "gsm8k",
        "reward_spec": {
            "method": "rule",
            "ground_truth": extract_solution(answer_raw),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
        # ✅ ALWAYS present
        "modalities": {"text_mod": []},
    }

    if add_modality:
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        data["modalities"] = {"text_mod": token_ids}

    return data



def extract_solution(solution_str: str) -> str:
    import re

    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def process_split(split_name: str, hf_split: str, tokenizer, placeholder: str, output_path: str):
    dataset = datasets.load_dataset("openai/gsm8k", "main")[hf_split]
    n = len(dataset)
    midpoint = n // 2

    def map_fn(example, idx):
        add_modality = idx < midpoint
        return build_sample(example, idx, split_name, tokenizer, placeholder, add_modality)

    processed = dataset.map(function=map_fn, with_indices=True)
    processed.to_parquet(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="~/data/gsm8k_modal_text")
    parser.add_argument("--placeholder_token", default="<|extra_0|>")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    train_out = os.path.join(output_dir, "train.parquet")
    val_out = os.path.join(output_dir, "validation.parquet")

    process_split("train", "train", tokenizer, args.placeholder_token, train_out)
    process_split("test", "test", tokenizer, args.placeholder_token, val_out)

    print(f"Wrote modal-text GSM8K to {train_out} and {val_out}")


if __name__ == "__main__":
    main()
