#!/usr/bin/env python
"""Fix dataset to have proper prompt format for TextWorld."""
import pandas as pd
from pathlib import Path

def fix_prompts():
    """Add proper prompt format that chat template can parse."""
    data_dir = Path("/home/parsaidp/data/textworld_memo")

    # For TextWorld, we need a minimal chat format prompt
    # The actual game interaction happens through the environment,
    # but the dataset loader expects a valid chat format
    dummy_prompt = [{"role": "user", "content": "Let's play a text adventure game."}]

    # Fix train
    train_df = pd.read_parquet(data_dir / "train.parquet")
    train_df['prompt'] = [dummy_prompt] * len(train_df)
    train_df.to_parquet(data_dir / "train.parquet", index=False)
    print(f"✓ Fixed train.parquet: {len(train_df)} samples")
    print(f"  Sample prompt: {train_df.iloc[0]['prompt']}")

    # Fix validation
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    val_df['prompt'] = [dummy_prompt] * len(val_df)
    val_df.to_parquet(data_dir / "validation.parquet", index=False)
    print(f"✓ Fixed validation.parquet: {len(val_df)} samples")

if __name__ == "__main__":
    fix_prompts()
