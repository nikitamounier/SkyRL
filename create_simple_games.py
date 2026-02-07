#!/usr/bin/env python
"""Create very simple TextWorld games using built-in challenges."""
import textworld
from pathlib import Path

def create_simple_games():
    """Create simple solvable games using TextWorld's challenge generator."""
    output_dir = Path("/home/parsaidp/data/textworld_memo/games_trivial")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Creating simple games using textworld-make...")

    # Create 10 very simple games with different seeds
    # These will be 1-room games with simple take/put tasks
    for i in range(1, 11):
        options = textworld.GameOptions()
        options.seeds = i * 123  # Different seed for variety
        options.nb_rooms = 1  # Single room
        options.nb_objects = 1  # Single object
        options.quest_length = 1  # 1-step solution
        options.quest_breadth = 1  # Linear quest

        game = textworld.generator.make_game(options)
        game_file = output_dir / f"simple_game_{i}.z8"
        game.save(str(game_file))
        print(f"✓ Created {game_file}")

    # Test one game
    print("\nTesting simple_game_1...")
    env = textworld.start(str(output_dir / "simple_game_1.z8"))
    game_state = env.reset()
    print(f"  Max score: {game_state.max_score}")
    print(f"  Observation: {game_state.feedback[:150]}...")

    # Try a few actions
    print("\n  Trying basic actions:")
    for cmd in ['look', 'inventory', 'examine table', 'take all']:
        game_state = env.reset()
        game_state, reward, done = env.step(cmd)
        if reward != 0 or done:
            print(f"    '{cmd}' -> reward={reward}, done={done}")

    env.close()
    return output_dir

if __name__ == "__main__":
    games_dir = create_simple_games()
    print(f"\nDone! Games saved to {games_dir}")
    print("\nTo train on these games, update the train.parquet file to point to these games.")
