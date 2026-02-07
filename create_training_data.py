#!/usr/bin/env python
"""Create training dataset with multiple simple games."""
import pandas as pd
from pathlib import Path

def create_multi_game_dataset():
    """Create train/val datasets using all simple games."""
    games_dir = Path("/home/parsaidp/data/textworld_memo/games_trivial")
    output_dir = Path("/home/parsaidp/data/textworld_memo")

    # Get all game files
    game_files = sorted(games_dir.glob("simple_game_*.z8"))
    print(f"Found {len(game_files)} games")

    # Create training data: use each game multiple times for more samples
    train_data = []
    for game_file in game_files[:8]:  # Use 8 games for training
        for repeat in range(5):  # 5 episodes per game = 40 training samples
            train_data.append({
                "game_file": str(game_file),
                "game_id": game_file.stem,
                "split": "train"
            })

    # Create validation data
    val_data = []
    for game_file in game_files[8:]:  # Use last 2 games for validation
        for repeat in range(2):  # 2 episodes per game = 4 validation samples
            val_data.append({
                "game_file": str(game_file),
                "game_id": game_file.stem,
                "split": "val"
            })

    # Save to parquet
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "validation.parquet", index=False)

    print(f"\n✓ Created training dataset:")
    print(f"  Train: {len(train_df)} samples across {len(train_df['game_id'].unique())} games")
    print(f"  Val: {len(val_df)} samples across {len(val_df['game_id'].unique())} games")
    print(f"\nDataset saved to {output_dir}")

    # Show sample
    print(f"\nSample train data:")
    print(train_df.head(10))

if __name__ == "__main__":
    create_multi_game_dataset()
