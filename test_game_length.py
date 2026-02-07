#!/usr/bin/env python3
"""Quick test to check game length and memory document creation."""

import sys
sys.path.insert(0, 'skyrl-gym')

from skyrl_gym.envs.textworld.env import TextWorldEnv
from omegaconf import DictConfig

def test_game(game_file, memory_window=3, game_name="SMALL"):
    """Test a game and report length + memory creation."""
    print(f"\n{'='*60}")
    print(f"Testing {game_name} game: {game_file}")
    print(f"Memory window: {memory_window} (doc at turns {memory_window}, {memory_window*2}, {memory_window*3}, ...)")
    print(f"{'='*60}\n")

    env_config = DictConfig({
        'max_turns': 50,
        'memory_window': memory_window,
        'max_memory_docs': 4,
        'tokenizer_path': 'Qwen/Qwen3-4B-Instruct-2507'
    })
    extras = {'game_file': game_file}
    env = TextWorldEnv(env_config, extras)

    # Initialize
    env.init([{'role': 'user', 'content': 'start'}])

    # Simple action sequence
    actions = ['look', 'inventory', 'examine room', 'go north', 'go south',
               'go east', 'go west', 'take all', 'drop all']

    prev_docs = 0
    for turn in range(1, 51):
        action = actions[turn % len(actions)]
        step_output = env.step(action)

        # Extract fields from BaseTextEnvStepOutput (TypedDict)
        obs = step_output['observations']
        reward = step_output['reward']
        done = step_output['done']

        # Check memory documents
        num_docs = len(env._memory.memory_documents) if hasattr(env, '_memory') else 0

        # Print turn info
        print(f"Turn {turn:2d}: ", end="")
        print(f"docs={num_docs}, reward={reward:5.2f}, done={str(done):5s}", end="")

        # Highlight new document creation
        if num_docs > prev_docs:
            print(" <- 🎯 NEW MEMORY DOCUMENT!")
            prev_docs = num_docs
        else:
            print()

        if done:
            print(f"\n✅ {game_name} game COMPLETED at turn {turn}")
            break
    else:
        print(f"\n⏱️  {game_name} game did NOT complete in {turn} turns")

    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total turns: {turn}")
    print(f"   Memory documents: {num_docs}")
    print(f"   History coverage: {num_docs * memory_window} turns")
    print(f"   Game completed: {done}")

    if turn < 15:
        print(f"   ⚠️  TOO SHORT - game ended before turn 15")
    elif turn < 25:
        print(f"   ⚠️  SHORT - consider MEDIUM games for better memory usage")
    else:
        print(f"   ✅ GOOD LENGTH - adequate for memory training")

    return turn, num_docs, done


if __name__ == "__main__":
    # Test SMALL game
    small_turns, small_docs, small_done = test_game(
        "/home/parsaidp/data/textworld_memo/games/game_small_1.z8",
        memory_window=3,
        game_name="SMALL"
    )

    # Test MEDIUM game if available
    try:
        medium_turns, medium_docs, medium_done = test_game(
            "/home/parsaidp/MeMo/tw_games_medium/game_medium_1.z8",
            memory_window=3,
            game_name="MEDIUM"
        )

        # Comparison
        print(f"\n{'='*60}")
        print(f"COMPARISON:")
        print(f"{'='*60}")
        print(f"SMALL:  {small_turns:2d} turns, {small_docs} docs")
        print(f"MEDIUM: {medium_turns:2d} turns, {medium_docs} docs")
        print(f"\nRecommendation: Use {'MEDIUM' if medium_turns > 25 else 'SMALL'} games")

    except Exception as e:
        print(f"\n⚠️  Could not test MEDIUM game: {e}")
        print(f"Recommendation: Generate MEDIUM games for better memory training")
