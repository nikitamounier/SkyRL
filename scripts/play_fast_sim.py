#!/usr/bin/env python3
"""
Interactive script to play TextWorld games using the FastTextWorldSimulator.

Usage:
    # Interactive play
    python scripts/play_fast_sim.py /path/to/game.json

    # Auto-play using the walkthrough
    python scripts/play_fast_sim.py /path/to/game.json --walkthrough

    # Benchmark: run all games with walkthroughs, report stats
    python scripts/play_fast_sim.py /path/to/games_dir --benchmark
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skyrl-gym"))

from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator


def interactive_play(game_json: str):
    """Play a game interactively in the terminal."""
    sim = FastTextWorldSimulator(game_json)
    obs, info = sim.reset()

    print(obs)
    print(f"\n[Score: {info['score']}/{info['max_score']}]")
    print("-" * 60)

    while True:
        action = input("\n> ").strip()
        if not action:
            continue
        if action.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        obs, reward, done, info = sim.step(action)
        print(obs)
        print(f"\n[Score: {info['score']}/{info['max_score']} | Reward: {reward}]")

        if done:
            if info.get("won"):
                print("\n*** You won! ***")
            else:
                print("\n*** Game over. ***")
            break


def walkthrough_play(game_json: str, verbose: bool = True):
    """Auto-play a game using the walkthrough from the JSON spec."""
    with open(game_json) as f:
        spec = json.load(f)

    walkthrough = spec.get("metadata", {}).get("walkthrough")
    if not walkthrough:
        print(f"No walkthrough found in {game_json}")
        return None

    sim = FastTextWorldSimulator(game_json)
    obs, info = sim.reset()

    if verbose:
        print(obs)
        print(f"[Score: {info['score']}/{info['max_score']}]")
        print(f"[Walkthrough: {len(walkthrough)} steps]")
        print("-" * 60)

    for i, action in enumerate(walkthrough):
        obs, reward, done, info = sim.step(action)
        if verbose:
            print(f"\n> {action}")
            print(obs)
            print(f"[Step {i+1}/{len(walkthrough)} | Score: {info['score']}/{info['max_score']} | Reward: {reward}]")

        if done:
            if verbose:
                if info.get("won"):
                    print("\n*** You won! ***")
                else:
                    print("\n*** Game over (not won). ***")
            break

    return {
        "game": os.path.basename(game_json),
        "steps": i + 1,
        "score": info["score"],
        "max_score": info["max_score"],
        "won": info.get("won", False),
        "done": done,
    }


def benchmark(games_dir: str):
    """Run all games in a directory with walkthroughs. Report stats."""
    json_files = sorted(glob.glob(os.path.join(games_dir, "*.json")))
    if not json_files:
        print(f"No .json game files found in {games_dir}")
        return

    print(f"Found {len(json_files)} games in {games_dir}")
    print("=" * 70)

    results = []
    total_steps = 0
    t0 = time.perf_counter()

    for gf in json_files:
        gt = time.perf_counter()
        result = walkthrough_play(gf, verbose=False)
        elapsed = time.perf_counter() - gt

        if result is None:
            print(f"  SKIP  {os.path.basename(gf):40s} (no walkthrough)")
            continue

        result["time_s"] = elapsed
        results.append(result)
        total_steps += result["steps"]

        status = "WON" if result["won"] else "FAIL"
        print(
            f"  {status:4s}  {result['game']:40s}  "
            f"score={result['score']}/{result['max_score']}  "
            f"steps={result['steps']:3d}  "
            f"time={elapsed:.3f}s"
        )

    total_time = time.perf_counter() - t0
    won = sum(1 for r in results if r["won"])

    print("=" * 70)
    print(f"Games: {len(results)}  |  Won: {won}/{len(results)}  |  Steps: {total_steps}")
    print(f"Total time: {total_time:.3f}s  |  Avg per game: {total_time/max(len(results),1):.3f}s")
    print(f"Steps/sec: {total_steps/max(total_time,0.001):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Play TextWorld games with FastTextWorldSimulator")
    parser.add_argument("path", help="Path to a .json game file or directory of games")
    parser.add_argument("--walkthrough", "-w", action="store_true", help="Auto-play using the walkthrough")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run all games in directory with walkthroughs")
    args = parser.parse_args()

    if args.benchmark:
        d = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
        benchmark(d)
    elif args.walkthrough:
        walkthrough_play(args.path)
    else:
        interactive_play(args.path)


if __name__ == "__main__":
    main()
