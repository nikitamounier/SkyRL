#!/usr/bin/env python3
"""
Play TextWorld games using OpenAI API to generate SFT/RL training data.

Features:
- Parallel game execution (--workers N)
- Multiple samples per game (--n_samples N) for diverse trajectories
- Saves per-game JSON + combined JSONL

Usage:
    # Fast: 8 parallel games
    python play_textworld_openai.py --data_file ~/data/textworld_memo/train.parquet --workers 8

    # Multiple samples per game for RL data
    python play_textworld_openai.py --data_file ~/data/textworld_memo/train.parquet --n_samples 3 --workers 8

    # Quick test
    python play_textworld_openai.py --data_file ~/data/textworld_memo/test.parquet --max_games 5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import textworld
from openai import OpenAI

_TW_LOCK = threading.Lock()


SYSTEM_PROMPT = (
    "You are playing a TextWorld text adventure game. The objective is stated at the "
    "start of the game — follow those instructions step by step.\n\n"
    "Valid commands:\n"
    "  Navigation: go north, go south, go east, go west\n"
    "  Items: take <item>, drop <item>, inventory, look\n"
    "  Interact: open <container>, close <container>, examine <object>\n"
    "  Keys: unlock <thing> with <key>, lock <thing> with <key>\n"
    "  Place: put <item> on <surface>, insert <item> into <container>\n\n"
    "Strategy:\n"
    "- Start with 'look' and 'inventory' to understand your situation\n"
    "- Examine objects and containers before moving on\n"
    "- Pick up keys and items — you'll need them later\n"
    "- Track which rooms you've visited to avoid going in circles\n"
    "- Follow the objective step by step\n\n"
    "Think about what you need to do next, then end your response with exactly one action:\n"
    "[ACTION: <command>]\n"
    "Only the text inside [ACTION: ...] is sent to the game."
)


def parse_action(text: str) -> str:
    if not text:
        return "look"
    match = re.search(r"\[ACTION:\s*([^\]]+)\]", text, re.IGNORECASE)
    if match:
        parsed = match.group(1).strip()
    else:
        cleaned = text.strip().split("\n")[-1].strip()
        parsed = cleaned
    try:
        parsed = parsed.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return "look"
    allowed = string.ascii_letters + string.digits + string.whitespace + "-_.,!?'"
    parsed = "".join(c for c in parsed if c in allowed)
    parsed = parsed.strip()[:200]
    return parsed if parsed else "look"


@dataclass
class GameTranscript:
    game_file: str
    game_id: str
    sample_id: int = 0
    turns: List[Dict[str, Any]] = field(default_factory=list)
    won: bool = False
    final_score: int = 0
    total_reward: float = 0.0
    num_turns: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)


def play_game(
    client: OpenAI,
    model: str,
    game_file: str,
    sample_id: int = 0,
    max_turns: int = 50,
    temperature: float = 1.0,
) -> GameTranscript:
    game_id = Path(game_file).stem
    transcript = GameTranscript(game_file=game_file, game_id=game_id, sample_id=sample_id)

    with _TW_LOCK:
        env = textworld.start(game_file)
    game_state = env.reset()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript.messages = messages

    for turn in range(1, max_turns + 1):
        obs = game_state.feedback.strip()
        if turn == 1:
            user_msg = f"You start a new game.\n\n{obs}"
        else:
            user_msg = obs

        messages.append({"role": "user", "content": user_msg})

        try:
            api_kwargs = dict(
                model=model,
                messages=messages,
                max_completion_tokens=1024,
            )
            if temperature != 1.0:
                api_kwargs["temperature"] = temperature
            response = client.chat.completions.create(**api_kwargs)
            assistant_text = response.choices[0].message.content or ""
        except Exception as e:
            assistant_text = "[ACTION: look]"

        action = parse_action(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})

        game_state, reward, done = env.step(action)
        score = getattr(game_state, "score", 0)
        won = bool(done and getattr(game_state, "won", False))

        transcript.turns.append({
            "turn": turn,
            "observation": obs,
            "action": action,
            "model_response": assistant_text,
            "reward": float(reward),
            "score": score,
            "done": done,
            "won": won,
        })
        transcript.total_reward += float(reward)
        transcript.final_score = score
        transcript.num_turns = turn

        if done:
            transcript.won = won
            break

    env.close()
    return transcript


def save_transcript(transcript: GameTranscript, output_dir: str, model: str):
    suffix = f"_s{transcript.sample_id}" if transcript.sample_id > 0 else ""
    out_file = os.path.join(output_dir, f"{transcript.game_id}{suffix}.json")
    with open(out_file, "w") as f:
        json.dump({
            "game_file": transcript.game_file,
            "game_id": transcript.game_id,
            "sample_id": transcript.sample_id,
            "model": model,
            "won": transcript.won,
            "final_score": transcript.final_score,
            "total_reward": transcript.total_reward,
            "num_turns": transcript.num_turns,
            "turns": transcript.turns,
            "messages": transcript.messages,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Play TextWorld with OpenAI API")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per game (>1 for diverse trajectories)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel game threads")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true", help="Skip games that already have output files")
    args = parser.parse_args()

    df = pd.read_parquet(args.data_file)
    game_files = df["game_file"].unique().tolist()
    if args.max_games:
        game_files = game_files[:args.max_games]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    client = OpenAI()

    # Build work items: (game_file, sample_id)
    work = []
    for gf in game_files:
        for s in range(args.n_samples):
            if args.skip_existing and args.output_dir:
                gid = Path(gf).stem
                suffix = f"_s{s}" if s > 0 else ""
                if os.path.exists(os.path.join(args.output_dir, f"{gid}{suffix}.json")):
                    continue
            work.append((gf, s))

    total = len(work)
    print(f"Model: {args.model}")
    print(f"Games: {len(game_files)}, Samples/game: {args.n_samples}")
    print(f"Total tasks: {total} ({total - len(work)} skipped)")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir or 'stdout only'}")
    print("=" * 70)

    completed = 0
    wins = 0
    t_start = time.time()

    def run_one(game_file, sample_id):
        return play_game(
            client=client,
            model=args.model,
            game_file=game_file,
            sample_id=sample_id,
            max_turns=args.max_turns,
            temperature=args.temperature,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, gf, s): (gf, s) for gf, s in work}

        for future in as_completed(futures):
            gf, s = futures[future]
            try:
                transcript = future.result()
                completed += 1
                if transcript.won:
                    wins += 1

                if args.output_dir:
                    save_transcript(transcript, args.output_dir, args.model)

                status = "WON" if transcript.won else f"score={transcript.final_score}"
                sample_str = f" s{s}" if args.n_samples > 1 else ""
                elapsed = time.time() - t_start
                rate = completed / elapsed * 3600 if elapsed > 0 else 0
                print(f"  [{completed}/{total}] {transcript.game_id}{sample_str}: {status} ({transcript.num_turns}t) [{rate:.0f}/hr]")
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total}] ERROR: {e}")

    elapsed = time.time() - t_start
    n = completed
    print(f"\n{'=' * 70}")
    print(f"RESULTS ({args.model}, {n} tasks, {elapsed:.0f}s)")
    print(f"  Win rate: {wins}/{n} ({100*wins/max(n,1):.1f}%)")
    print(f"  Rate: {n/elapsed*3600:.0f} games/hr")

    if args.output_dir:
        summary = {
            "model": args.model,
            "data_file": args.data_file,
            "num_tasks": n,
            "wins": wins,
            "win_rate": wins / max(n, 1),
            "elapsed_seconds": elapsed,
            "max_turns": args.max_turns,
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "workers": args.workers,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        jsonl_file = os.path.join(args.output_dir, "transcripts.jsonl")
        all_files = sorted(Path(args.output_dir).glob("game_*.json"))
        with open(jsonl_file, "w") as f:
            for gf in all_files:
                with open(gf) as gfh:
                    data = json.load(gfh)
                f.write(json.dumps({
                    "game_id": data["game_id"],
                    "sample_id": data.get("sample_id", 0),
                    "won": data["won"],
                    "score": data["final_score"],
                    "turns": data["turns"],
                    "messages": data["messages"],
                }) + "\n")
        print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
