#!/usr/bin/env python3
"""
Evaluate Qwen3 on TextWorld dataset.

Modes:
  - Baseline (no memory): default
  - Text memory: --memory_window N injects memory document text into the system prompt
    every N turns, matching the SkyRL env's _IncrementalMemory logic exactly.

Batched inference: all active games generate in a single vLLM call each turn.

Usage:
    python eval_textworld_baseline.py --data_file ~/data/textworld_memo/test.parquet
    python eval_textworld_baseline.py --data_file ~/data/textworld_memo/test.parquet --memory_window 3
    python eval_textworld_baseline.py --data_file ~/data/textworld_memo/test.parquet --memory_window 5
"""

from __future__ import annotations

import argparse
import json
import re
import string
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import textworld
from vllm import LLM, SamplingParams


_TW_LOCK = threading.Lock()


def parse_action(action: str) -> str:
    """Parse action — matches SkyRL TextWorldEnv._parse_action()."""
    if not action:
        return "look"
    try:
        action = action.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return "look"
    if not action:
        return "look"
    match = re.search(r"\[ACTION:\s*([^\]]+)\]", action, re.IGNORECASE)
    if match:
        parsed = match.group(1).strip()
    else:
        cleaned = action.replace("<think>", "").replace("</think>", "").strip()
        if "</think>" in action:
            _, after = action.split("</think>", 1)
            cleaned = after.strip() or cleaned
        for prefix in ("Action:", "I will", "I'll", "Let me", "I would", "I should"):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[0].strip()
        parsed = cleaned.strip("\"'.")
    allowed = string.ascii_letters + string.digits + string.whitespace + "-_.,!?'"
    parsed = "".join(c for c in parsed if c in allowed)
    return parsed.strip()[:200] or "look"


def create_memory_document(segment: List[Dict]) -> str:
    """Create memory document text — matches SkyRL _IncrementalMemory._create_document_text()."""
    if not segment:
        return ""
    start_turn = segment[0]["turn"]
    end_turn = segment[-1]["turn"]
    total_reward = sum(t["reward"] for t in segment)
    score_delta = segment[-1]["score"] - segment[0]["score"]
    lines = [f"Turns {start_turn}-{end_turn} | Reward: {total_reward} | Score Change: {score_delta}", ""]
    for t in segment:
        lines.append(f"Turn {t['turn']}:")
        if t["observation"]:
            lines.append(f"Obs: {t['observation'][:200]}")
        action_line = f"Act: {t['action']}"
        if t["reward"] != 0:
            action_line += f" -> Reward: {t['reward']}"
        lines.append(action_line)
        lines.append("")
    return "\n".join(lines).strip()


@dataclass
class GameState:
    """Tracks state for one active game."""
    idx: int
    game_file: str
    system_prompt: str
    memory_window: int = 0  # 0 = no memory
    max_memory_docs: int = 20
    env: Any = None
    tw_state: Any = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False
    won: bool = False
    final_score: int = 0
    num_turns: int = 0
    # Memory tracking
    current_segment: List[Dict] = field(default_factory=list)
    memory_documents: List[str] = field(default_factory=list)


def main():
    parser = argparse.ArgumentParser(description="TextWorld evaluation (batched)")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_generate_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Memory options
    parser.add_argument("--memory_window", type=int, default=0,
                        help="Create memory doc every N turns (0=no memory, matches SkyRL env)")
    parser.add_argument("--max_memory_docs", type=int, default=20)
    args = parser.parse_args()

    mode_label = f"memory_w{args.memory_window}" if args.memory_window > 0 else "baseline"

    # Resolve local snapshot
    if args.model_path == "Qwen/Qwen3-4B-Instruct-2507":
        cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
        if cache_dir.exists():
            snapshots = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                args.model_path = str(snapshots[0])
                print(f"Using local snapshot: {args.model_path}")

    # Load dataset
    import datasets
    ds = datasets.Dataset.from_parquet(args.data_file)
    print(f"Loaded {len(ds)} rows from {args.data_file}")

    seen, eval_rows = set(), []
    for row in ds:
        gf = row["game_file"]
        if gf not in seen:
            seen.add(gf)
            eval_rows.append(row)
    print(f"Unique games: {len(eval_rows)}")

    if args.max_games:
        eval_rows = eval_rows[:args.max_games]
        print(f"Evaluating first {len(eval_rows)} games")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    print(f"Mode: {mode_label}")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        seed=args.seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_generate_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    all_results = []
    batch_size = args.batch_size
    total_games = len(eval_rows)

    print(f"\nEvaluating {total_games} games (max_turns={args.max_turns}, batch={batch_size}, {mode_label})")
    print("=" * 70)

    t_start = time.time()

    for batch_start in range(0, total_games, batch_size):
        batch_rows = eval_rows[batch_start:batch_start + batch_size]
        print(f"\nBatch {batch_start // batch_size + 1}/{(total_games + batch_size - 1) // batch_size}: "
              f"games {batch_start+1}-{batch_start+len(batch_rows)}")

        games: List[GameState] = []
        for i, row in enumerate(batch_rows):
            prompt_msgs = row["prompt"]
            system_prompt = prompt_msgs[0]["content"] if prompt_msgs else (
                "You are playing a text-based adventure game. "
                "Respond with a single action command."
            )
            gs = GameState(
                idx=batch_start + i,
                game_file=row["game_file"],
                system_prompt=system_prompt,
                memory_window=args.memory_window,
                max_memory_docs=args.max_memory_docs,
                messages=[{"role": "system", "content": system_prompt}],
            )
            with _TW_LOCK:
                gs.env = textworld.start(row["game_file"])
            gs.tw_state = gs.env.reset()
            games.append(gs)

        for turn in range(1, args.max_turns + 1):
            active = [g for g in games if not g.done]
            if not active:
                break

            prompts = []
            for g in active:
                obs = g.tw_state.feedback.strip()

                # If memory enabled, update system message with memory docs
                if g.memory_window > 0 and g.memory_documents:
                    memory_text = "\n\n---\n".join(g.memory_documents)
                    g.messages[0]["content"] = (
                        g.system_prompt + "\n\nMemory from previous turns:\n" + memory_text
                    )

                g.messages.append({"role": "user", "content": obs})

                prompt_str = tokenizer.apply_chat_template(
                    g.messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_str)

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for g, output in zip(active, outputs):
                raw_response = output.outputs[0].text
                action = parse_action(raw_response)

                g.actions.append(action)
                g.messages.append({"role": "assistant", "content": raw_response})
                g.num_turns = len(g.actions)

                # Step env
                g.tw_state, reward, done = g.env.step(action)
                g.total_reward += float(reward)
                g.final_score = getattr(g.tw_state, "score", 0)
                g.won = bool(done and getattr(g.tw_state, "won", False))
                g.done = done or turn >= args.max_turns

                # Memory tracking (if enabled)
                if g.memory_window > 0:
                    g.current_segment.append({
                        "turn": turn,
                        "observation": g.tw_state.feedback.strip() if not g.done else "",
                        "action": action,
                        "reward": float(reward),
                        "score": g.final_score,
                    })
                    if len(g.current_segment) >= g.memory_window:
                        doc = create_memory_document(g.current_segment)
                        if doc:
                            g.memory_documents.append(doc)
                            if len(g.memory_documents) > g.max_memory_docs:
                                g.memory_documents = g.memory_documents[-g.max_memory_docs:]
                        g.current_segment = []
                    if g.done and g.current_segment:
                        doc = create_memory_document(g.current_segment)
                        if doc:
                            g.memory_documents.append(doc)
                        g.current_segment = []

        for g in games:
            g.env.close()
            result = {
                "game_file": g.game_file,
                "won": g.won,
                "final_score": g.final_score,
                "total_reward": g.total_reward,
                "num_turns": g.num_turns,
                "num_memory_docs": len(g.memory_documents),
                "messages": g.messages,  # full conversation (for trajectory inspection)
            }
            all_results.append(result)
            status = "WON" if g.won else f"score={g.final_score}"
            docs_str = f", {len(g.memory_documents)} docs" if g.memory_window > 0 else ""
            print(f"  [{g.idx+1}/{total_games}] {Path(g.game_file).stem}: {status} ({g.num_turns} turns{docs_str})")

    elapsed = time.time() - t_start

    n = len(all_results)
    wins = sum(1 for r in all_results if r["won"])
    total_score = sum(r["final_score"] for r in all_results)
    total_reward = sum(r["total_reward"] for r in all_results)

    print(f"\n{'=' * 70}")
    print(f"RESULTS — {mode_label} ({n} games, {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Solve rate: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Avg score:  {total_score/n:.2f}")
    print(f"  Avg reward: {total_reward/n:.2f}")
    avg_turns = sum(r["num_turns"] for r in all_results) / n
    print(f"  Avg turns:  {avg_turns:.1f}")
    won_turns = [r["num_turns"] for r in all_results if r["won"]]
    lost_turns = [r["num_turns"] for r in all_results if not r["won"]]
    if won_turns:
        print(f"  Avg turns (won):  {sum(won_turns)/len(won_turns):.1f}")
    if lost_turns:
        print(f"  Avg turns (lost): {sum(lost_turns)/len(lost_turns):.1f}")

    by_diff: Dict[str, list] = {}
    for r in all_results:
        name = Path(r["game_file"]).stem
        diff = "hard" if "_hard_" in name else ("medium" if "_medium_" in name else "unknown")
        by_diff.setdefault(diff, []).append(r)

    if len(by_diff) > 1:
        print(f"\nBy difficulty:")
        for diff in ("medium", "hard", "unknown"):
            if diff not in by_diff:
                continue
            dr = by_diff[diff]
            dw = sum(1 for r in dr if r["won"])
            print(f"  {diff}: {dw}/{len(dr)} solved ({100*dw/len(dr):.1f}%), avg_score={sum(r['final_score'] for r in dr)/len(dr):.2f}")

    summary = {
        "model": args.model_path,
        "mode": mode_label,
        "data_file": args.data_file,
        "memory_window": args.memory_window,
        "num_games": n,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "elapsed_seconds": elapsed,
        "solve_rate": wins / n if n else 0,
        "avg_score": total_score / n if n else 0,
        "avg_reward": total_reward / n if n else 0,
        "avg_turns": avg_turns,
        "avg_turns_won": sum(won_turns) / len(won_turns) if won_turns else None,
        "avg_turns_lost": sum(lost_turns) / len(lost_turns) if lost_turns else None,
        "by_difficulty": {
            diff: {
                "num_games": len(dr),
                "wins": sum(1 for r in dr if r["won"]),
                "solve_rate": sum(1 for r in dr if r["won"]) / len(dr) if dr else 0,
                "avg_score": sum(r["final_score"] for r in dr) / len(dr) if dr else 0,
            }
            for diff, dr in by_diff.items()
        },
        "results": all_results,
    }

    split_name = Path(args.data_file).stem
    output_file = args.output_file or str(
        Path(args.data_file).parent / f"eval_results/{mode_label}_{split_name}.json"
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
