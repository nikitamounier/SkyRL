#!/usr/bin/env python
"""Fix dataset format to include required 'prompt' column."""
import pandas as pd
from pathlib import Path

def fix_dataset():
    """Add dummy 'prompt' column that TextWorld doesn't actually use."""
    data_dir = Path("/home/parsaidp/data/textworld_memo")

    # Fix train
    train_df = pd.read_parquet(data_dir / "train.parquet")
    train_df['prompt'] = ""  # Dummy prompt - TextWorld uses game files instead
    train_df.to_parquet(data_dir / "train.parquet", index=False)
    print(f"✓ Fixed train.parquet: {len(train_df)} samples")
    print(f"  Columns: {train_df.columns.tolist()}")

    # Fix validation
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    val_df['prompt'] = ""  # Dummy prompt
    val_df.to_parquet(data_dir / "validation.parquet", index=False)
    print(f"✓ Fixed validation.parquet: {len(val_df)} samples")
    print(f"  Columns: {val_df.columns.tolist()}")

if __name__ == "__main__":
    fix_dataset()
