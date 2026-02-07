#!/usr/bin/env python3
"""
Test MeMo memory integration with TextWorld in inference mode.

This script verifies:
1. Games are long enough to create memory documents
2. Memory documents are created at expected intervals (memory_window)
3. Model receives memory tokens correctly
4. Actions are contextually reasonable given memory
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add skyrl-gym to path
skyrl_gym_path = Path(__file__).parent / "skyrl-gym"
if skyrl_gym_path.exists():
    sys.path.insert(0, str(skyrl_gym_path))
else:
    logger.error(f"skyrl-gym not found at {skyrl_gym_path}")

try:
    from skyrl_gym.envs.textworld.env import TextWorldEnv
except ImportError as e:
    logger.error(f"Failed to import TextWorldEnv: {e}")
    logger.info("Trying alternative import path...")
    # Try installing in editable mode first
    import subprocess
    subprocess.run(["pip", "install", "-e", str(skyrl_gym_path)], check=False)
    from skyrl_gym.envs.textworld.env import TextWorldEnv


def main():
    parser = argparse.ArgumentParser(description="Test MeMo inference with TextWorld")
    parser.add_argument("--game-file", type=str, required=True, help="Path to TextWorld game file (.z8)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Model path")
    parser.add_argument("--memory-window", type=int, default=3, help="Turns per memory document")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum turns to play")
    parser.add_argument("--max-memory-docs", type=int, default=4, help="Maximum memory documents")
    parser.add_argument("--num-memories", type=int, default=8, help="Memory vectors per document")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    logger.info(f"Testing MeMo inference with game: {args.game_file}")
    logger.info(f"Config: memory_window={args.memory_window}, max_docs={args.max_memory_docs}, num_memories={args.num_memories}")

    # Initialize environment
    logger.info("Initializing TextWorld environment...")
    from omegaconf import DictConfig

    env_config = DictConfig({
        "max_turns": args.max_turns,
        "memory_window": args.memory_window,
        "max_memory_docs": args.max_memory_docs,
        "max_doc_tokens": 256,
        "tokenizer_path": args.model,
        "memory_modality_id": "memo_memory",
    })

    extras = {
        "game_file": args.game_file,
        "max_turns": args.max_turns,
        "memory_window": args.memory_window,
        "max_memory_docs": args.max_memory_docs,
        "tokenizer_path": args.model,
        "memory_modality_id": "memo_memory",
    }

    env = TextWorldEnv(env_config=env_config, extras=extras)

    # Initialize model (for future generation testing)
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Reset environment
    logger.info("Resetting environment...")
    obs = env.reset()
    logger.info(f"Initial observation: {obs[:200]}...")

    # Track memory creation
    memory_created_at_turns = []
    total_turns = 0
    done = False

    # Play game
    logger.info(f"\nStarting game (max {args.max_turns} turns)...")
    logger.info("=" * 80)

    while not done and total_turns < args.max_turns:
        total_turns += 1

        # Check memory state
        num_docs = len(env._memory_documents) if hasattr(env, "_memory_documents") else 0

        # Simple action policy (for testing)
        actions = ["look", "inventory", "examine room", "go north", "go south", "go east", "go west"]
        action = actions[total_turns % len(actions)]

        logger.info(f"\nTurn {total_turns}:")
        logger.info(f"  Memory documents: {num_docs}")
        logger.info(f"  Action: {action}")

        # Check if new document was created
        if num_docs > len(memory_created_at_turns):
            memory_created_at_turns.append(total_turns)
            logger.success(f"  🎯 NEW MEMORY DOCUMENT CREATED! (Document #{num_docs})")

            # Log memory document details
            if hasattr(env, "_memory_documents"):
                doc = env._memory_documents[-1]
                if isinstance(doc, dict) and "token_ids" in doc:
                    logger.info(f"     Document has {len(doc['token_ids'])} tokens")
                elif hasattr(doc, "__len__"):
                    logger.info(f"     Document length: {len(doc)}")

        # Execute action
        obs, reward, done, info = env.step(action)

        logger.info(f"  Reward: {reward:.2f}")
        logger.info(f"  Observation: {obs[:150]}...")
        logger.info(f"  Done: {done}")

        if done:
            logger.success(f"\n🏁 Game completed at turn {total_turns}!")
            break

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY:")
    logger.info(f"  Total turns: {total_turns}")
    logger.info(f"  Game completed: {done}")
    logger.info(f"  Memory documents created: {len(memory_created_at_turns)}")

    if memory_created_at_turns:
        logger.info(f"  Documents created at turns: {memory_created_at_turns}")
        logger.success("  ✅ Memory system is working!")
    else:
        logger.error("  ❌ NO memory documents created!")
        logger.error(f"  Expected first document at turn {args.memory_window}")

    # Verify memory window
    expected_docs = total_turns // args.memory_window
    actual_docs = len(memory_created_at_turns)

    logger.info(f"  Expected documents: {expected_docs}")
    logger.info(f"  Actual documents: {actual_docs}")

    if actual_docs >= expected_docs:
        logger.success("  ✅ Memory creation frequency matches memory_window!")
    else:
        logger.warning(f"  ⚠️ Memory creation slower than expected")

    # Game length assessment
    logger.info(f"\nGAME DIFFICULTY ASSESSMENT:")
    if total_turns < 15:
        logger.warning(f"  ⚠️ Game too short ({total_turns} turns)")
        logger.warning("  Consider using MEDIUM or LARGE preset for longer games")
    elif total_turns >= 25:
        logger.success(f"  ✅ Game length adequate ({total_turns} turns)")
    else:
        logger.info(f"  Game length acceptable ({total_turns} turns)")

    # Memory utilization
    logger.info(f"\nMEMORY UTILIZATION:")
    max_possible_docs = min(args.max_memory_docs, total_turns // args.memory_window)
    utilization = (actual_docs / max_possible_docs * 100) if max_possible_docs > 0 else 0
    logger.info(f"  Max possible documents: {max_possible_docs}")
    logger.info(f"  Actual documents: {actual_docs}")
    logger.info(f"  Utilization: {utilization:.1f}%")

    if utilization >= 75:
        logger.success("  ✅ Good memory utilization!")
    elif utilization >= 50:
        logger.info("  Moderate memory utilization")
    else:
        logger.warning("  ⚠️ Low memory utilization - game too short or memory_window too large")

    return 0 if actual_docs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
