#!/usr/bin/env python3
"""
Hybrid data generation — two stages:

  Stage 1 (GPU, no internet): Qwen3 plays TextWorld games via vLLM.
    Saves full transcripts with memory document snapshots at every turn.

  Stage 2 (CPU/login, internet): GPT oracle annotates transcripts.
    For every turn that has memory documents, GPT predicts the best action.
    Saves oracle samples as SFT training labels.

Usage:
    # Stage 1: Qwen3 plays on GPU node
    python play_textworld_hybrid.py stage1 \
        --data_file ~/data/textworld_memo/train.parquet \
        --memory_window 5 --batch_size 20 \
        --output_dir /path/to/output

    # Stage 2: GPT oracle on login node (has internet)
    OPENAI_API_KEY=sk-... python play_textworld_hybrid.py stage2 \
        --transcript_dir /path/to/output \
        --gpt_model gpt-5-mini --gpt_workers 8
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
    if game_file.endswith(".z8"):
        return game_file[:-3] + ".json"
    if game_file.endswith(".json"):
        return game_file
    return game_file + ".json"


# ===================================================================
# STAGE 1: Qwen3 plays games on GPU (no internet needed)
# ===================================================================

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
    memory_window: int = 5
    max_memory_docs: int = 20
    current_segment: List[Dict] = field(default_factory=list)
    memory_documents: List[str] = field(default_factory=list)


def run_stage1(args):
    """Qwen3 plays games via vLLM. Saves transcripts with memory snapshots."""
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

    total_games = len(game_files)
    print(f"\nGames: {total_games}")
    print(f"Memory window: {args.memory_window}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    total_wins = 0
    games_done = 0
    t_start = time.time()

    for batch_start in range(0, total_games, args.batch_size):
        batch_files = game_files[batch_start:batch_start + args.batch_size]
        print(f"\nBatch {batch_start // args.batch_size + 1}/{(total_games + args.batch_size - 1) // args.batch_size}: "
              f"games {batch_start+1}-{batch_start+len(batch_files)}")

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

        for turn in range(1, args.max_turns + 1):
            active = [g for g in games if not g.done]
            if not active:
                break

            # Qwen3 batch inference
            prompts = []
            for g in active:
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

            for g, output in zip(active, outputs):
                raw_response = output.outputs[0].text
                action = parse_action(raw_response)
                current_obs = g.messages[-1]["content"]

                obs, reward, done, info = g.sim.step(action)
                score = info.get("score", 0)
                won = info.get("won", False)

                # Snapshot memory docs at this turn BEFORE updating them
                memory_snapshot = list(g.memory_documents)

                g.turns.append({
                    "turn": turn,
                    "observation": current_obs,
                    "action": action,
                    "model_response": raw_response,
                    "reward": float(reward),
                    "score": score,
                    "done": done,
                    "won": won,
                    "next_observation": obs.strip(),
                    "memory_documents_snapshot": memory_snapshot,
                })
                g.total_reward += float(reward)
                g.final_score = score
                g.won = bool(done and won)
                g.done = done or turn >= args.max_turns
                g.num_turns = turn

                g.messages.append({"role": "assistant", "content": raw_response})

                # Memory tracking
                g.current_segment.append({
                    "turn": turn,
                    "observation": current_obs[:200],
                    "action": action,
                    "reward": float(reward),
                    "score": score,
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

                if not g.done:
                    g.messages.append({"role": "user", "content": obs.strip()})

        # Save transcripts
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
                "memory_window": args.memory_window,
                "num_memory_docs": len(g.memory_documents),
                "turns": g.turns,
                "memory_documents": g.memory_documents,
            }
            out_file = os.path.join(args.output_dir, f"{g.game_id}.json")
            with open(out_file, "w") as f:
                json.dump(transcript, f, indent=2)

            status = "WON" if g.won else f"score={g.final_score}"
            elapsed = time.time() - t_start
            rate = games_done / elapsed * 3600 if elapsed > 0 else 0
            print(f"  [{games_done}/{total_games}] {g.game_id}: {status} "
                  f"({g.num_turns}t, {len(g.memory_documents)} docs) [{rate:.0f}/hr]")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"STAGE 1 COMPLETE ({total_games} games, {elapsed:.0f}s)")
    print(f"  Win rate: {total_wins}/{total_games} ({100*total_wins/max(total_games,1):.1f}%)")
    print(f"  Transcripts saved to: {args.output_dir}")
    print(f"\nNext: run stage2 from a node with internet access:")
    print(f"  python play_textworld_hybrid.py stage2 --transcript_dir {args.output_dir}")


# ===================================================================
# STAGE 2: GPT oracle annotates transcripts (needs internet)
# ===================================================================

def call_gpt_oracle(
    client,
    model: str,
    game_id: str,
    turn_num: int,
    memory_documents: List[str],
    conversation_history: List[Dict],
    observation: str,
) -> Dict[str, Any]:
    """Call GPT API to predict best action at a given game state."""
    memory_text = ""
    if memory_documents:
        memory_text = "\n\nMemory from previous turns:\n" + "\n\n---\n".join(memory_documents)

    gpt_messages = [
        {"role": "system", "content": GPT_ORACLE_SYSTEM_PROMPT + memory_text}
    ]
    gpt_messages.extend(conversation_history)
    gpt_messages.append({"role": "user", "content": observation})

    gpt_response = ""
    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=gpt_messages,
                temperature=1.0,
                max_completion_tokens=4096,
            )
            gpt_response = response.choices[0].message.content or ""
            if gpt_response:
                break
        except Exception as e:
            last_error = e
            print(f"    [GPT ERROR] {game_id} turn {turn_num} attempt {attempt+1}/3: {type(e).__name__}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue
    if not gpt_response:
        if last_error:
            print(f"    [GPT FAILED] {game_id} turn {turn_num}: all 3 attempts failed")
        gpt_response = f"[ERROR: {last_error}]\n[ACTION: look]"

    return {
        "gpt_response": gpt_response,
        "gpt_action": parse_action(gpt_response),
        "prompt": gpt_messages,
    }


def run_stage2(args):
    """GPT oracle annotates Qwen3 transcripts with best-action predictions."""
    from openai import OpenAI
    client = OpenAI()

    # Find transcript files
    transcript_dir = args.transcript_dir
    transcript_files = sorted(Path(transcript_dir).glob("game_*.json"))
    if not transcript_files:
        print(f"No transcript files found in {transcript_dir}")
        return

    if args.max_games:
        transcript_files = transcript_files[:args.max_games]

    print(f"GPT oracle: {args.gpt_model} (workers={args.gpt_workers})")
    print(f"Transcripts: {len(transcript_files)} games in {transcript_dir}")
    print("=" * 70)

    all_oracle_samples = []
    all_sft_samples = []
    total_calls = 0
    t_start = time.time()

    for tf_idx, tf in enumerate(transcript_files):
        with open(tf) as f:
            transcript = json.load(f)

        game_id = transcript["game_id"]
        turns = transcript["turns"]

        # Build oracle requests: every turn that has memory documents
        oracle_requests = []
        conversation_history = []

        for turn_data in turns:
            memory_docs = turn_data.get("memory_documents_snapshot", [])

            if memory_docs and not turn_data.get("done", False):
                oracle_requests.append({
                    "game_id": game_id,
                    "turn": turn_data["turn"],
                    "memory_documents": memory_docs,
                    "conversation_history": list(conversation_history),
                    "observation": turn_data.get("next_observation", ""),
                    "qwen3_action": turn_data["action"],
                    "qwen3_response": turn_data.get("model_response", ""),
                    "score": turn_data["score"],
                    "reward": turn_data["reward"],
                })

            # Build up conversation history for subsequent turns
            conversation_history.append({"role": "user", "content": turn_data["observation"]})
            conversation_history.append({"role": "assistant", "content": f"[ACTION: {turn_data['action']}]"})

        if not oracle_requests:
            print(f"  [{tf_idx+1}/{len(transcript_files)}] {game_id}: 0 oracle turns (skipped)")
            continue

        # Call GPT in parallel
        game_samples = []
        with ThreadPoolExecutor(max_workers=args.gpt_workers) as pool:
            futures = {}
            for req in oracle_requests:
                fut = pool.submit(
                    call_gpt_oracle,
                    client,
                    args.gpt_model,
                    req["game_id"],
                    req["turn"],
                    req["memory_documents"],
                    req["conversation_history"],
                    req["observation"],
                )
                futures[fut] = req

            for fut in as_completed(futures):
                req = futures[fut]
                try:
                    result = fut.result()
                    sample = {
                        "game_id": req["game_id"],
                        "turn": req["turn"],
                        "memory_documents": req["memory_documents"],
                        "num_memory_docs": len(req["memory_documents"]),
                        "prompt": result["prompt"],
                        "gpt_response": result["gpt_response"],
                        "gpt_action": result["gpt_action"],
                        "qwen3_action": req["qwen3_action"],
                        "qwen3_response": req["qwen3_response"],
                        "score": req["score"],
                        "reward": req["reward"],
                        "game_won": transcript.get("won", False),
                    }
                    game_samples.append(sample)
                    all_oracle_samples.append(sample)
                    total_calls += 1

                    # SFT-format sample with full conversation history
                    if not result["gpt_response"].startswith("[ERROR"):
                        memory_docs = req["memory_documents"]
                        sys_prompt = QWEN_SYSTEM_PROMPT
                        if memory_docs:
                            memory_text = "\n\nMemory from previous turns:\n" + "\n\n---\n".join(memory_docs)
                            sys_prompt = QWEN_SYSTEM_PROMPT + memory_text

                        sft_messages = [{"role": "system", "content": sys_prompt}]
                        sft_messages.extend(req["conversation_history"])
                        sft_messages.append({"role": "user", "content": req["observation"]})
                        sft_messages.append({"role": "assistant", "content": f"[ACTION: {result['gpt_action']}]"})

                        sft_sample = {
                            "messages": sft_messages,
                            "game_id": req["game_id"],
                            "turn": req["turn"],
                            "source": "hybrid_oracle",
                        }
                        all_sft_samples.append(sft_sample)

                except Exception as e:
                    print(f"    GPT error for {req['game_id']} turn {req['turn']}: {e}")

        # Update transcript with oracle samples
        transcript["oracle_samples"] = sorted(game_samples, key=lambda s: s["turn"])
        transcript["num_oracle_samples"] = len(game_samples)
        with open(tf, "w") as f:
            json.dump(transcript, f, indent=2)

        elapsed = time.time() - t_start
        valid = sum(1 for s in game_samples if not s["gpt_response"].startswith("[ERROR"))
        print(f"  [{tf_idx+1}/{len(transcript_files)}] {game_id}: "
              f"{valid}/{len(game_samples)} oracle OK "
              f"({'WON' if transcript.get('won') else 'lost'}) "
              f"[{(tf_idx+1)/elapsed*3600:.0f}/hr]")

    # Save combined oracle JSONL
    oracle_file = os.path.join(transcript_dir, "oracle_samples.jsonl")
    with open(oracle_file, "w") as f:
        for sample in sorted(all_oracle_samples, key=lambda s: (s["game_id"], s["turn"])):
            f.write(json.dumps(sample) + "\n")

    # Save SFT JSONL
    sft_file = os.path.join(transcript_dir, "hybrid_sft.jsonl")
    with open(sft_file, "w") as f:
        for sample in all_sft_samples:
            f.write(json.dumps(sample) + "\n")

    elapsed = time.time() - t_start
    valid_total = sum(1 for s in all_oracle_samples if not s["gpt_response"].startswith("[ERROR"))
    print(f"\n{'=' * 70}")
    print(f"STAGE 2 COMPLETE ({len(transcript_files)} games, {elapsed:.0f}s)")
    print(f"  Oracle calls: {total_calls} ({valid_total} valid)")
    print(f"  Oracle samples: {oracle_file}")
    print(f"  SFT samples:    {sft_file} ({len(all_sft_samples)} samples)")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Qwen3 + GPT oracle data generation")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    # Stage 1: Qwen3 plays
    p1 = subparsers.add_parser("stage1", help="Qwen3 plays games on GPU (no internet needed)")
    p1.add_argument("--data_file", type=str, required=True)
    p1.add_argument("--output_dir", type=str, required=True)
    p1.add_argument("--max_games", type=int, default=None)
    p1.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p1.add_argument("--max_turns", type=int, default=50)
    p1.add_argument("--max_generate_length", type=int, default=256)
    p1.add_argument("--temperature", type=float, default=0.6)
    p1.add_argument("--top_p", type=float, default=0.95)
    p1.add_argument("--batch_size", type=int, default=20)
    p1.add_argument("--memory_window", type=int, default=5)
    p1.add_argument("--max_memory_docs", type=int, default=20)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--skip_existing", action="store_true")

    # Stage 2: GPT oracle
    p2 = subparsers.add_parser("stage2", help="GPT oracle annotates transcripts (needs internet)")
    p2.add_argument("--transcript_dir", type=str, required=True)
    p2.add_argument("--max_games", type=int, default=None)
    p2.add_argument("--gpt_model", type=str, default="gpt-5-mini")
    p2.add_argument("--gpt_workers", type=int, default=8)

    args = parser.parse_args()

    if args.stage == "stage1":
        run_stage1(args)
    elif args.stage == "stage2":
        run_stage2(args)


if __name__ == "__main__":
    main()
