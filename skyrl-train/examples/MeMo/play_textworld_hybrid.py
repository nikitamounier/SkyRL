#!/usr/bin/env python3
"""
Hybrid data generation: Qwen3 plays TextWorld games (vLLM), GPT predicts
best actions at memory steps (when memory documents accumulate).

Qwen3's actions drive the game. GPT's predictions are saved as SFT labels
on Qwen3's actual state distribution.

Usage:
    # Full run
    python play_textworld_hybrid.py \
        --data_file ~/data/textworld_memo/train.parquet \
        --model_path Qwen/Qwen3-4B-Instruct-2507 \
        --gpt_model gpt-5-mini \
        --memory_window 5 --batch_size 20 --gpt_workers 8 \
        --output_dir /path/to/output

    # Quick test (no GPU needed — uses fast sim only with GPT oracle)
    python play_textworld_hybrid.py \
        --data_file ~/data/textworld_memo/test.parquet \
        --max_games 3 --memory_window 5 \
        --output_dir /tmp/hybrid_test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add skyrl-gym to path for FastTextWorldSimulator
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "skyrl-gym"))
from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator

from openai import OpenAI


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

QWEN_SYSTEM_PROMPT = (
    "You are playing a TextWorld text adventure game. The objective is stated at the "
    "start of the game — follow those instructions step by step.\n\n"
    "Valid commands:\n"
    "  Navigation: go north, go south, go east, go west\n"
    "  Items: take <item>, drop <item>, inventory, look\n"
    "  Interact: open <container>, close <container>, examine <object>\n"
    "  Keys: unlock <thing> with <key>, lock <thing> with <key>\n"
    "  Place: put <item> on <surface>, insert <item> into <container>\n\n"
    "Think about what you need to do next, then end your response with exactly one action:\n"
    "[ACTION: <command>]\n"
    "Only the text inside [ACTION: ...] is sent to the game."
)

GPT_ORACLE_SYSTEM_PROMPT = (
    "You are an expert TextWorld player. Given the game history and memory of past turns, "
    "predict the single best next action.\n\n"
    "Valid commands:\n"
    "  Navigation: go north, go south, go east, go west\n"
    "  Items: take <item>, drop <item>, inventory, look\n"
    "  Interact: open <container>, close <container>, examine <object>\n"
    "  Keys: unlock <thing> with <key>, lock <thing> with <key>\n"
    "  Place: put <item> on <surface>, insert <item> into <container>\n\n"
    "Think step by step about the objective, what has been done, and what remains. "
    "Then output exactly one action:\n"
    "[ACTION: <command>]"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_action(text: str) -> str:
    if not text:
        return "look"
    try:
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return "look"
    match = re.search(r"\[ACTION:\s*([^\]]+)\]", text, re.IGNORECASE)
    if match:
        parsed = match.group(1).strip()
    else:
        cleaned = text.replace("<think>", "").replace("</think>", "").strip()
        if "</think>" in text:
            _, after = text.split("</think>", 1)
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
    """Create memory document text — matches SkyRL _IncrementalMemory."""
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


def resolve_game_json(game_file: str) -> str:
    """Resolve .z8 path to .json for FastTextWorldSimulator."""
    if game_file.endswith(".z8"):
        return game_file[:-3] + ".json"
    if game_file.endswith(".json"):
        return game_file
    return game_file + ".json"


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    idx: int
    game_file: str
    game_json: str
    game_id: str
    sim: Optional[FastTextWorldSimulator] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    won: bool = False
    final_score: int = 0
    num_turns: int = 0
    total_reward: float = 0.0
    # Memory tracking
    memory_window: int = 5
    max_memory_docs: int = 20
    current_segment: List[Dict] = field(default_factory=list)
    memory_documents: List[str] = field(default_factory=list)
    # Oracle data — collected at memory steps
    oracle_samples: List[Dict[str, Any]] = field(default_factory=list)
    # Flag: this game needs GPT oracle call this turn
    pending_oracle: bool = False
    pending_oracle_obs: str = ""


# ---------------------------------------------------------------------------
# GPT oracle call
# ---------------------------------------------------------------------------

def call_gpt_oracle(
    client: OpenAI,
    model: str,
    game: GameState,
    observation: str,
) -> Dict[str, Any]:
    """Call GPT API with game history + memory to predict best action."""
    # Build GPT messages: oracle system prompt + memory + conversation history
    memory_text = ""
    if game.memory_documents:
        memory_text = "\n\nMemory from previous turns:\n" + "\n\n---\n".join(game.memory_documents)

    gpt_messages = [
        {"role": "system", "content": GPT_ORACLE_SYSTEM_PROMPT + memory_text}
    ]

    # Add conversation history (user observations + assistant actions)
    for turn_data in game.turns:
        gpt_messages.append({"role": "user", "content": turn_data["observation"]})
        gpt_messages.append({"role": "assistant", "content": f"[ACTION: {turn_data['action']}]"})

    # Add current observation
    gpt_messages.append({"role": "user", "content": observation})

    gpt_response = ""
    # Retry up to 3 times — GPT-5-mini intermittently returns empty content
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=gpt_messages,
                temperature=1.0,
                max_completion_tokens=512,
            )
            gpt_response = response.choices[0].message.content or ""
            if gpt_response:
                break
        except Exception as e:
            gpt_response = f"[ERROR: {e}]\n[ACTION: look]"
            break

    gpt_action = parse_action(gpt_response)

    return {
        "game_id": game.game_id,
        "turn": game.num_turns,
        "memory_documents": list(game.memory_documents),
        "num_memory_docs": len(game.memory_documents),
        "prompt": gpt_messages,
        "gpt_response": gpt_response,
        "gpt_action": gpt_action,
        "qwen3_action": game.turns[-1]["action"] if game.turns else "",
        "qwen3_response": game.turns[-1].get("model_response", "") if game.turns else "",
        "score": game.final_score,
        "reward": game.turns[-1]["reward"] if game.turns else 0.0,
        "game_won": game.won,
        "game_done": game.done,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hybrid Qwen3 + GPT oracle data generation")
    # Data
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    # Qwen3 (vLLM)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_generate_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=20)
    # GPT oracle
    parser.add_argument("--gpt_model", type=str, default="gpt-5-mini")
    parser.add_argument("--gpt_workers", type=int, default=8)
    # Memory
    parser.add_argument("--memory_window", type=int, default=5)
    parser.add_argument("--max_memory_docs", type=int, default=20)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_parquet(args.data_file)
    game_files = df["game_file"].unique().tolist()
    if args.max_games:
        game_files = game_files[:args.max_games]

    if args.skip_existing:
        before = len(game_files)
        game_files = [
            gf for gf in game_files
            if not os.path.exists(os.path.join(args.output_dir, f"{Path(gf).stem}.json"))
        ]
        print(f"Skipping {before - len(game_files)} existing games")

    # Resolve local model snapshot
    model_path = args.model_path
    if model_path == "Qwen/Qwen3-4B-Instruct-2507":
        cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
        if cache_dir.exists():
            snapshots = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                model_path = str(snapshots[0])
                print(f"Using local snapshot: {model_path}")

    # Load vLLM
    from vllm import LLM, SamplingParams

    print(f"\nLoading Qwen3: {model_path}")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        max_model_len=32768,
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

    # OpenAI client for GPT oracle
    gpt_client = OpenAI()

    total_games = len(game_files)
    print(f"\nGames: {total_games}")
    print(f"Qwen3: {model_path} (batch={args.batch_size})")
    print(f"GPT oracle: {args.gpt_model} (workers={args.gpt_workers})")
    print(f"Memory window: {args.memory_window}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    all_oracle_samples = []
    total_oracle_calls = 0
    total_wins = 0
    t_start = time.time()
    games_done = 0

    for batch_start in range(0, total_games, args.batch_size):
        batch_files = game_files[batch_start:batch_start + args.batch_size]
        print(f"\nBatch {batch_start // args.batch_size + 1}/{(total_games + args.batch_size - 1) // args.batch_size}: "
              f"games {batch_start+1}-{batch_start+len(batch_files)}")

        # Initialize games
        games: List[GameState] = []
        for i, gf in enumerate(batch_files):
            game_json = resolve_game_json(gf)
            gs = GameState(
                idx=batch_start + i,
                game_file=gf,
                game_json=game_json,
                game_id=Path(gf).stem,
                memory_window=args.memory_window,
                max_memory_docs=args.max_memory_docs,
                messages=[{"role": "system", "content": QWEN_SYSTEM_PROMPT}],
            )
            gs.sim = FastTextWorldSimulator(game_json)
            obs, info = gs.sim.reset()
            gs.messages.append({"role": "user", "content": obs.strip()})
            games.append(gs)

        # Turn loop
        for turn in range(1, args.max_turns + 1):
            active = [g for g in games if not g.done]
            if not active:
                break

            # --- Qwen3 batch inference ---
            prompts = []
            for g in active:
                # Inject memory into system message if available
                if g.memory_documents:
                    memory_text = "\n\nMemory from previous turns:\n" + "\n\n---\n".join(g.memory_documents)
                    g.messages[0]["content"] = QWEN_SYSTEM_PROMPT + memory_text
                else:
                    g.messages[0]["content"] = QWEN_SYSTEM_PROMPT

                prompt_str = tokenizer.apply_chat_template(
                    g.messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_str)

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            # --- Process Qwen3 responses + step games ---
            for g, output in zip(active, outputs):
                raw_response = output.outputs[0].text
                action = parse_action(raw_response)

                # Current observation (the last user message)
                current_obs = g.messages[-1]["content"]

                # Step the game with Qwen3's action
                obs, reward, done, info = g.sim.step(action)
                score = info.get("score", 0)
                won = info.get("won", False)

                g.turns.append({
                    "turn": turn,
                    "observation": current_obs,
                    "action": action,
                    "model_response": raw_response,
                    "reward": float(reward),
                    "score": score,
                    "done": done,
                    "won": won,
                })
                g.total_reward += float(reward)
                g.final_score = score
                g.won = bool(done and won)
                g.done = done or turn >= args.max_turns
                g.num_turns = turn

                # Append Qwen3 response to conversation
                g.messages.append({"role": "assistant", "content": raw_response})

                # Memory tracking
                g.current_segment.append({
                    "turn": turn,
                    "observation": current_obs[:200],
                    "action": action,
                    "reward": float(reward),
                    "score": score,
                })

                g.pending_oracle = False
                if len(g.current_segment) >= g.memory_window:
                    doc = create_memory_document(g.current_segment)
                    if doc:
                        g.memory_documents.append(doc)
                        if len(g.memory_documents) > g.max_memory_docs:
                            g.memory_documents = g.memory_documents[-g.max_memory_docs:]
                    g.current_segment = []
                    # Mark for GPT oracle if game is still going
                    if not g.done:
                        g.pending_oracle = True
                        g.pending_oracle_obs = obs.strip()

                # Finalize memory on game end
                if g.done and g.current_segment:
                    doc = create_memory_document(g.current_segment)
                    if doc:
                        g.memory_documents.append(doc)
                    g.current_segment = []

                # Add next observation to conversation if game continues
                if not g.done:
                    g.messages.append({"role": "user", "content": obs.strip()})

            # --- GPT oracle calls for games at memory steps ---
            oracle_games = [g for g in active if g.pending_oracle]
            if oracle_games:
                with ThreadPoolExecutor(max_workers=args.gpt_workers) as pool:
                    futures = {
                        pool.submit(
                            call_gpt_oracle,
                            gpt_client,
                            args.gpt_model,
                            g,
                            g.pending_oracle_obs,
                        ): g
                        for g in oracle_games
                    }
                    for future in as_completed(futures):
                        g = futures[future]
                        try:
                            sample = future.result()
                            g.oracle_samples.append(sample)
                            all_oracle_samples.append(sample)
                            total_oracle_calls += 1
                        except Exception as e:
                            print(f"    GPT oracle error for {g.game_id}: {e}")

                        g.pending_oracle = False

        # --- Save per-game transcripts ---
        for g in games:
            games_done += 1
            if g.won:
                total_wins += 1

            transcript = {
                "game_file": g.game_file,
                "game_id": g.game_id,
                "won": g.won,
                "final_score": g.final_score,
                "total_reward": g.total_reward,
                "num_turns": g.num_turns,
                "num_oracle_samples": len(g.oracle_samples),
                "num_memory_docs": len(g.memory_documents),
                "turns": g.turns,
                "oracle_samples": g.oracle_samples,
                "memory_documents": g.memory_documents,
            }
            out_file = os.path.join(args.output_dir, f"{g.game_id}.json")
            with open(out_file, "w") as f:
                json.dump(transcript, f, indent=2)

            status = "WON" if g.won else f"score={g.final_score}"
            oracle_str = f", {len(g.oracle_samples)} oracle" if g.oracle_samples else ""
            elapsed = time.time() - t_start
            rate = games_done / elapsed * 3600 if elapsed > 0 else 0
            print(f"  [{games_done}/{total_games}] {g.game_id}: {status} "
                  f"({g.num_turns}t{oracle_str}) [{rate:.0f}/hr]")

    # --- Save combined oracle JSONL ---
    oracle_file = os.path.join(args.output_dir, "oracle_samples.jsonl")
    with open(oracle_file, "w") as f:
        for sample in all_oracle_samples:
            f.write(json.dumps(sample) + "\n")

    # --- Summary ---
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_games} games, {elapsed:.0f}s)")
    print(f"  Win rate: {total_wins}/{total_games} ({100*total_wins/max(total_games,1):.1f}%)")
    print(f"  Oracle calls: {total_oracle_calls}")
    print(f"  Oracle samples saved: {oracle_file}")
    print(f"  Rate: {total_games/elapsed*3600:.0f} games/hr")

    summary = {
        "qwen3_model": args.model_path,
        "gpt_model": args.gpt_model,
        "data_file": args.data_file,
        "num_games": total_games,
        "wins": total_wins,
        "win_rate": total_wins / max(total_games, 1),
        "total_oracle_calls": total_oracle_calls,
        "memory_window": args.memory_window,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
