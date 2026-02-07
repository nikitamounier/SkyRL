#!/usr/bin/env python3
"""Create parquet files from existing games."""
import pandas as pd
import json
from pathlib import Path

games_dir = Path("/home/parsaidp/data/textworld_memo_long/games")
game_files = sorted(games_dir.glob("*.z8"))

print(f"Found {len(game_files)} game files")

data = []
for game_file in game_files:
    # Extract metadata from filename: game_000_q4r3.z8
    parts = game_file.stem.split('_')
    quest_info = parts[-1]  # q4r3
    quest_length = int(quest_info.split('r')[0][1:])  # Extract 4 from q4

    sample = {
        'prompt': [
            {"role": "system", "content": "You are playing a text adventure game."},
            {"role": "user", "content": "Game initialized."}
        ],
        'env_class': 'textworld',
        'game_file': str(game_file),
        'max_score': 1,  # Will be determined at runtime
        'expected_turns': quest_length + 2
    }
    data.append(sample)

df = pd.DataFrame(data)

# Training set (90%)
train_df = df.sample(frac=0.9, random_state=42)
train_file = "/home/parsaidp/data/textworld_memo_long/train.parquet"
train_df.to_parquet(train_file)
print(f"✓ Saved {len(train_df)} training samples to {train_file}")

# Validation set (10%)
val_df = df.drop(train_df.index)
val_file = "/home/parsaidp/data/textworld_memo_long/validation.parquet"
val_df.to_parquet(val_file)
print(f"✓ Saved {len(val_df)} validation samples to {val_file}")

print("\n✅ Ready to train!")
print(f"To submit training: DATA_DIR=/home/parsaidp/data/textworld_memo_long sbatch train_textworld_memo.slurm")
