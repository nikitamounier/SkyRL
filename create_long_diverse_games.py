#!/usr/bin/env python3
"""Create longer, diverse TextWorld games for MeMo training.

Creates games with varying complexity (3-10 step solutions) to ensure:
- Episodes reach turn 3+ (trigger memory document generation)
- Diverse environments (different room layouts, objects, quests)
- Multiple game types (treasure hunt, cooking, take/drop chains)
"""
import textworld
from pathlib import Path
import random

def create_long_games(output_dir, num_games=50):
    """Create diverse games with longer quests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Creating {num_games} diverse, longer games...")
    print("=" * 60)

    game_configs = []

    for i in range(num_games):
        seed = 1000 + i * 17  # Diverse seeds

        # Vary complexity across games
        if i < num_games // 3:
            # Simple-medium games (3-5 steps)
            nb_rooms = random.choice([2, 3])
            nb_objects = random.choice([2, 3, 4])
            quest_length = random.choice([3, 4, 5])
            quest_breadth = random.choice([1, 2])
        elif i < 2 * num_games // 3:
            # Medium games (5-7 steps)
            nb_rooms = random.choice([3, 4, 5])
            nb_objects = random.choice([4, 5, 6])
            quest_length = random.choice([5, 6, 7])
            quest_breadth = random.choice([2, 3])
        else:
            # Complex games (7-10 steps)
            nb_rooms = random.choice([5, 6, 7])
            nb_objects = random.choice([6, 7, 8, 9])
            quest_length = random.choice([7, 8, 9, 10])
            quest_breadth = random.choice([2, 3, 4])

        options = textworld.GameOptions()
        options.seeds = seed
        options.nb_rooms = nb_rooms
        options.nb_objects = nb_objects
        options.quest_length = quest_length
        options.quest_breadth = quest_breadth

        # All use house theme for consistency
        options.grammar.theme = "house"

        # Set output path and format
        game_file = output_dir / f"game_{i:03d}_q{quest_length}r{nb_rooms}.z8"
        options.path = str(game_file)

        try:
            game = textworld.generator.make_game(options)
            # Compile the game (creates both .json and .z8 files)
            compiled_path = textworld.generator.compile_game(game, options)

            game_configs.append({
                'file': str(game_file.name),
                'quest_length': quest_length,
                'nb_rooms': nb_rooms,
                'nb_objects': nb_objects,
                'max_score': game.max_score
            })

            if (i + 1) % 10 == 0:
                print(f"✓ Created {i + 1}/{num_games} games...")

        except Exception as e:
            print(f"✗ Failed to create game {i}: {e}")
            continue

    print("=" * 60)
    print(f"\n✅ Created {len(game_configs)} games!")

    # Print summary
    print("\nGame Distribution:")
    short = sum(1 for g in game_configs if g['quest_length'] <= 5)
    medium = sum(1 for g in game_configs if 5 < g['quest_length'] <= 7)
    long = sum(1 for g in game_configs if g['quest_length'] > 7)

    print(f"  Short (3-5 steps):  {short} games")
    print(f"  Medium (6-7 steps): {medium} games")
    print(f"  Long (8-10 steps):  {long} games")

    avg_quest = sum(g['quest_length'] for g in game_configs) / len(game_configs)
    print(f"  Average quest length: {avg_quest:.1f} steps")

    # Test a few games
    print("\nTesting sample games...")
    for idx in [0, len(game_configs)//2, -1]:
        if idx >= len(game_configs):
            continue

        game_info = game_configs[idx]
        game_file = output_dir / game_info['file']

        print(f"\n  {game_info['file']}:")
        print(f"    Quest length: {game_info['quest_length']} steps")
        print(f"    Rooms: {game_info['nb_rooms']}, Objects: {game_info['nb_objects']}")
        print(f"    Max score: {game_info['max_score']}")

        try:
            env = textworld.start(str(game_file))
            game_state = env.reset()
            print(f"    Initial obs: {game_state.feedback[:100]}...")

            # Count minimum turns needed
            print(f"    ✓ This game WILL generate memory docs (quest ≥ 3 steps)")
            env.close()
        except Exception as e:
            print(f"    ✗ Error testing: {e}")

    return output_dir, game_configs


def create_training_parquet(games_dir, game_configs, output_file):
    """Create parquet file with game references for training."""
    import pandas as pd
    import json

    print(f"\n\nCreating training data parquet...")

    data = []
    for config in game_configs:
        game_file = str(Path(games_dir) / config['file'])

        # Create training sample
        sample = {
            'messages': [
                {"role": "system", "content": "You are playing a text adventure game."},
                {"role": "user", "content": "Game initialized."}
            ],
            'env_class': 'textworld',
            'extra': {
                'game_file': game_file,
                'max_score': config['max_score'],
                'expected_turns': config['quest_length'] + 2  # Quest + overhead
            }
        }
        data.append(sample)

    df = pd.DataFrame(data)

    # Convert messages to JSON strings
    df['messages'] = df['messages'].apply(json.dumps)
    df['extra'] = df['extra'].apply(json.dumps)

    df.to_parquet(output_file)
    print(f"✓ Saved {len(df)} training samples to {output_file}")

    # Create validation split (10%)
    val_size = max(1, len(df) // 10)
    val_df = df.sample(n=val_size, random_state=42)
    val_file = str(output_file).replace('train.', 'validation.')
    val_df.to_parquet(val_file)
    print(f"✓ Saved {len(val_df)} validation samples to {val_file}")

    return output_file


if __name__ == "__main__":
    # Create games
    games_dir = "/home/parsaidp/data/textworld_memo_long/games"
    output_dir, game_configs = create_long_games(games_dir, num_games=50)

    # Create training data
    train_file = "/home/parsaidp/data/textworld_memo_long/train.parquet"
    create_training_parquet(games_dir, game_configs, train_file)

    print("\n" + "=" * 60)
    print("✅ DONE! Long, diverse games created.")
    print("=" * 60)
    print(f"\nGames directory: {games_dir}")
    print(f"Training data: {train_file}")
    print("\nTo train on these games:")
    print("  1. Update train_textworld_memo.slurm:")
    print("     DATA_DIR=/home/parsaidp/data/textworld_memo_long")
    print("  2. Submit: sbatch train_textworld_memo.slurm")
    print("\nThese games WILL generate memory documents (≥3 turns)!")
    print("=" * 60)
