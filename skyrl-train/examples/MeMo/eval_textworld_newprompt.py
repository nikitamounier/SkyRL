#!/usr/bin/env python3
"""Eval Qwen3-4B on TextWorld with the improved system prompt (same as GPT-5-mini test)."""

import argparse, json, re, string, time, threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd
import textworld
from vllm import LLM, SamplingParams

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


def parse_action(text):
    if not text:
        return "look"
    match = re.search(r"\[ACTION:\s*([^\]]+)\]", text, re.IGNORECASE)
    if match:
        parsed = match.group(1).strip()
    else:
        parsed = text.strip().split("\n")[-1].strip()
    try:
        parsed = parsed.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return "look"
    allowed = string.ascii_letters + string.digits + string.whitespace + "-_.,!?'"
    parsed = "".join(c for c in parsed if c in allowed).strip()[:200]
    return parsed if parsed else "look"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--max_generate_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu_memory", type=float, default=0.9)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_file)
    game_files = df["game_file"].unique().tolist()
    if args.max_games:
        game_files = game_files[:args.max_games]

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        max_tokens=args.max_generate_length,
        temperature=args.temperature,
        top_p=0.9,
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_path}")
    print(f"Games: {len(game_files)}, Max turns: {args.max_turns}")
    print("=" * 70)

    results = []
    t0 = time.time()

    for gi, gf in enumerate(game_files):
        gid = Path(gf).stem
        with _TW_LOCK:
            env = textworld.start(gf)
        state = env.reset()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        won = False
        score = 0

        for turn in range(1, args.max_turns + 1):
            obs = state.feedback.strip()
            if turn == 1:
                messages.append({"role": "user", "content": f"You start a new game.\n\n{obs}"})
            else:
                messages.append({"role": "user", "content": obs})

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = llm.generate([prompt], sampling, use_tqdm=False)
            text = output[0].outputs[0].text
            messages.append({"role": "assistant", "content": text})

            action = parse_action(text)
            state, reward, done = env.step(action)
            score = getattr(state, "score", 0)
            won = bool(done and getattr(state, "won", False))

            if done:
                break

        env.close()
        status = "WON" if won else f"score={score}"
        print(f"  [{gi+1}/{len(game_files)}] {gid}: {status} ({turn} turns)")
        results.append({"game": gid, "won": won, "score": score, "turns": turn})

    elapsed = time.time() - t0
    wins = sum(r["won"] for r in results)
    n = len(results)
    print(f"\n{'='*70}")
    print(f"RESULTS ({n} games, {elapsed:.0f}s)")
    print(f"  Solve rate: {wins}/{n} ({100*wins/max(n,1):.1f}%)")
    print(f"  Avg score:  {sum(r['score'] for r in results)/max(n,1):.2f}")
    print(f"  Avg turns:  {sum(r['turns'] for r in results)/max(n,1):.1f}")

    if args.output_dir:
        with open(Path(args.output_dir) / "results.json", "w") as f:
            json.dump({"results": results, "wins": wins, "total": n, "elapsed": elapsed}, f, indent=2)


if __name__ == "__main__":
    main()
