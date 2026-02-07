#!/usr/bin/env python
"""Update TextWorld dataset prompts with proper instructions for action format."""
import pandas as pd
from pathlib import Path

def update_prompts():
    """Add instructions for outputting actions in parseable format."""
    data_dir = Path("/home/parsaidp/data/textworld_memo")

    # New prompt with clear instructions
    system_prompt = {
        "role": "system",
        "content": """You are playing a text adventure game. You can think through your strategy, but you must always end your response with a clear action command.

Format your responses like this:
1. Think about what you observed and what to do next
2. End with [ACTION: command] where command is a simple verb or verb+object

Example:
I see a chest drawer in the bedroom. I should try to open it to find items.
[ACTION: open chest drawer]

Valid action verbs: look, examine, take, drop, open, close, go, inventory, eat, use
Keep actions simple and clear."""
    }

    user_prompt = {
        "role": "user",
        "content": "Let's play a text adventure game."
    }

    new_prompt = [system_prompt, user_prompt]

    # Fix train
    train_df = pd.read_parquet(data_dir / "train.parquet")
    train_df['prompt'] = [new_prompt] * len(train_df)
    train_df.to_parquet(data_dir / "train.parquet", index=False)
    print(f"✓ Updated train.parquet: {len(train_df)} samples")
    print(f"  Sample prompt: {train_df.iloc[0]['prompt']}")

    # Fix validation
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    val_df['prompt'] = [new_prompt] * len(val_df)
    val_df.to_parquet(data_dir / "validation.parquet", index=False)
    print(f"✓ Updated validation.parquet: {len(val_df)} samples")

if __name__ == "__main__":
    update_prompts()
