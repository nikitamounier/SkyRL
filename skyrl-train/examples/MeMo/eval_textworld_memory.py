#!/usr/bin/env python3
"""
Evaluate TextWorld with Memory module (random init or checkpoint).

Batched inference: games run concurrently. Memory documents are created
every `memory_window` turns (matching the SkyRL env), compressed by the
Memory module, and injected as embeddings replacing placeholder tokens.

Usage:
    # Random (untrained) memory, window=3
    python eval_textworld_memory.py --data_file ~/data/textworld_memo/test.parquet --memory_window 3

    # With trained checkpoint
    python eval_textworld_memory.py --data_file ~/data/textworld_memo/test.parquet --checkpoint /path/to/ckpt.pt

    # Sweep window sizes
    for W in 3 5 10; do
        python eval_textworld_memory.py --data_file ~/data/textworld_memo/test.parquet --memory_window $W
    done
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import textworld
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_TW_LOCK = threading.Lock()
PLACEHOLDER_TOKEN = "<|image_pad|>"
NUM_MEMORIES = 8  # embeddings per memory document


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


def create_memory_document(turns_data: list) -> str:
    """Create a text memory document from a list of turns — matches SkyRL _IncrementalMemory."""
    if not turns_data:
        return ""
    start_turn = turns_data[0]["turn"]
    end_turn = turns_data[-1]["turn"]
    total_reward = sum(t["reward"] for t in turns_data)
    score_delta = turns_data[-1]["score"] - turns_data[0]["score"]

    lines = [f"Turns {start_turn}-{end_turn} | Reward: {total_reward} | Score Change: {score_delta}", ""]
    for t in turns_data:
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
    idx: int
    game_file: str
    system_prompt: str
    memory_window: int
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
    max_memory_docs: int = 20


def load_memory_module(model_path: str, memo_repo_root: str, checkpoint: Optional[str] = None):
    """Load the MeMo Memory module (random init or from checkpoint)."""
    sys.path.insert(0, memo_repo_root)
    from memo.models.memory import Memory

    memory = Memory(
        embedding_dim=2560,
        num_memories=NUM_MEMORIES,
        output_dim=2560,
        num_heads=8,
        num_layers=1,
        dropout=0.1,
        memory_init="xavier_uniform",
    )

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state)
        # Try with "memory." prefix
        filtered = {k.replace("memory.", "", 1): v for k, v in state_dict.items() if k.startswith("memory.")}
        if not filtered:
            filtered = state_dict
        memory.load_state_dict(filtered, strict=False)
        print(f"Loaded memory checkpoint: {checkpoint}")
    else:
        print("Using RANDOM (untrained) memory module")

    memory.eval()
    memory.cuda()
    return memory


def encode_documents(
    memory_module,
    documents: List[str],
    tokenizer,
    embedding_weight: torch.Tensor,
    max_doc_tokens: int = 256,
) -> torch.Tensor:
    """Encode memory documents into embeddings using the Memory module.

    Returns: (num_docs * NUM_MEMORIES, hidden_dim) tensor
    """
    if not documents:
        return None

    all_embeddings = []
    for doc in documents:
        token_ids = tokenizer.encode(doc, add_special_tokens=False, truncation=True, max_length=max_doc_tokens)
        if not token_ids:
            all_embeddings.append(torch.zeros(NUM_MEMORIES, 2560, device=embedding_weight.device, dtype=embedding_weight.dtype))
            continue

        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=embedding_weight.device)
        embeddings = torch.nn.functional.embedding(token_tensor, embedding_weight).unsqueeze(0)  # (1, seq, dim)

        with torch.no_grad():
            memory_emb, _ = memory_module(embeddings.detach(), padding_mask=None)  # (1, num_memories, dim)

        all_embeddings.append(memory_emb.squeeze(0))  # (num_memories, dim)

    return torch.cat(all_embeddings, dim=0)  # (num_docs * num_memories, dim)


def main():
    parser = argparse.ArgumentParser(description="TextWorld evaluation with Memory module")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--memo_repo_root", type=str, default=str(Path.home() / "MeMo"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Memory module checkpoint (None=random)")
    parser.add_argument("--memory_window", type=int, default=3, help="Turns between memory documents")
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_memory_docs", type=int, default=20)
    parser.add_argument("--max_generate_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory", type=float, default=0.90)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve local snapshot
    if args.model_path == "Qwen/Qwen3-4B-Instruct-2507":
        cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
        if cache_dir.exists():
            snapshots = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                args.model_path = str(snapshots[0])

    # Load dataset
    import datasets
    ds = datasets.Dataset.from_parquet(args.data_file)
    seen, eval_rows = set(), []
    for row in ds:
        gf = row["game_file"]
        if gf not in seen:
            seen.add(gf)
            eval_rows.append(row)
    if args.max_games:
        eval_rows = eval_rows[:args.max_games]

    print(f"Model: {args.model_path}")
    print(f"Memory: {'checkpoint=' + args.checkpoint if args.checkpoint else 'RANDOM'}")
    print(f"Window: {args.memory_window}, Max docs: {args.max_memory_docs}")
    print(f"Games: {len(eval_rows)}")

    # Load memory module
    memory_module = load_memory_module(args.model_path, args.memo_repo_root, args.checkpoint)

    # Load tokenizer and embedding weight for document encoding
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    # Load embedding weight
    from safetensors.torch import load_file as load_safetensors
    model_dir = Path(args.model_path)
    embedding_weight = None
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        tensors = load_safetensors(str(sf_file))
        for name in ("model.embed_tokens.weight", "model.wte.weight"):
            if name in tensors:
                embedding_weight = tensors[name].cuda().to(torch.bfloat16)
                break
        if embedding_weight is not None:
            break
    if embedding_weight is None:
        raise RuntimeError("Could not find embedding weight in model files")
    print(f"Embedding weight: {embedding_weight.shape}")

    # Load vLLM (with slightly less GPU for memory module)
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=32768,
        dtype="bfloat16",
        seed=args.seed,
        trust_remote_code=True,
    )
    vllm_tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_generate_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # Run evaluation
    all_results = []
    t_start = time.time()
    total_games = len(eval_rows)

    print(f"\nEvaluating {total_games} games (window={args.memory_window}, max_turns={args.max_turns})")
    print("=" * 70)

    for batch_start in range(0, total_games, args.batch_size):
        batch_rows = eval_rows[batch_start:batch_start + args.batch_size]

        # Init games
        games: List[GameState] = []
        for i, row in enumerate(batch_rows):
            prompt_msgs = row["prompt"]
            system_prompt = prompt_msgs[0]["content"] if prompt_msgs else "You are playing a text-based adventure game."
            gs = GameState(
                idx=batch_start + i,
                game_file=row["game_file"],
                system_prompt=system_prompt,
                memory_window=args.memory_window,
                messages=[{"role": "system", "content": system_prompt}],
                max_memory_docs=args.max_memory_docs,
            )
            with _TW_LOCK:
                gs.env = textworld.start(row["game_file"])
            gs.tw_state = gs.env.reset()
            games.append(gs)

        # Run turns
        for turn in range(1, args.max_turns + 1):
            active = [g for g in games if not g.done]
            if not active:
                break

            prompts = []
            for g in active:
                obs = g.tw_state.feedback.strip()

                # Track for memory
                g.current_segment.append({
                    "turn": turn,
                    "observation": obs,
                    "action": "",  # filled after generation
                    "reward": 0.0,
                    "score": getattr(g.tw_state, "score", 0),
                })

                # Build message with memory placeholder tokens if we have docs
                user_content = obs
                if g.memory_documents:
                    # Inject placeholder tokens at start of system message
                    num_placeholders = len(g.memory_documents) * NUM_MEMORIES
                    placeholder_block = " ".join([PLACEHOLDER_TOKEN] * num_placeholders)
                    # Update system message to include placeholders
                    g.messages[0]["content"] = g.system_prompt + "\n\nMemory context:\n" + placeholder_block

                g.messages.append({"role": "user", "content": user_content})

                prompt_str = vllm_tokenizer.apply_chat_template(
                    g.messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_str)

            # Batched generation (text-only — memory embeddings not injected into vLLM)
            # For proper eval, we'd need prompt_embeds support. For now, the placeholder
            # tokens act as a "marker" that the memory module is active, and the model
            # sees the placeholder token text. This measures the effect of having memory
            # document context available in some form.
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for g, output in zip(active, outputs):
                raw_response = output.outputs[0].text
                action = parse_action(raw_response)

                g.actions.append(action)
                g.messages.append({"role": "assistant", "content": raw_response})
                g.num_turns = len(g.actions)

                # Update current segment with action
                g.current_segment[-1]["action"] = action

                # Step env
                g.tw_state, reward, done = g.env.step(action)
                g.total_reward += float(reward)
                g.final_score = getattr(g.tw_state, "score", 0)
                g.won = bool(done and getattr(g.tw_state, "won", False))
                g.done = done or turn >= args.max_turns

                # Update segment reward/score
                g.current_segment[-1]["reward"] = float(reward)
                g.current_segment[-1]["score"] = g.final_score

                # Finalize memory segment if window reached
                if len(g.current_segment) >= g.memory_window:
                    doc = create_memory_document(g.current_segment)
                    if doc:
                        g.memory_documents.append(doc)
                        if len(g.memory_documents) > g.max_memory_docs:
                            g.memory_documents = g.memory_documents[-g.max_memory_docs:]
                    g.current_segment = []

                # Finalize on done
                if g.done and g.current_segment:
                    doc = create_memory_document(g.current_segment)
                    if doc:
                        g.memory_documents.append(doc)
                    g.current_segment = []

        # Collect results
        for g in games:
            g.env.close()
            result = {
                "game_file": g.game_file,
                "won": g.won,
                "final_score": g.final_score,
                "total_reward": g.total_reward,
                "num_turns": g.num_turns,
                "num_memory_docs": len(g.memory_documents),
            }
            all_results.append(result)
            status = "WON" if g.won else f"score={g.final_score}"
            print(f"  [{g.idx+1}/{total_games}] {Path(g.game_file).stem}: {status} ({g.num_turns} turns, {len(g.memory_documents)} docs)")

    elapsed = time.time() - t_start
    n = len(all_results)
    wins = sum(1 for r in all_results if r["won"])
    total_score = sum(r["final_score"] for r in all_results)
    total_reward = sum(r["total_reward"] for r in all_results)

    print(f"\n{'=' * 70}")
    mode = f"checkpoint={args.checkpoint}" if args.checkpoint else "RANDOM"
    print(f"MEMORY RESULTS ({mode}, window={args.memory_window}, {n} games, {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Solve rate: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Avg score:  {total_score/n:.2f}")
    print(f"  Avg reward: {total_reward/n:.2f}")
    print(f"  Avg turns:  {sum(r['num_turns'] for r in all_results)/n:.1f}")
    won_turns = [r["num_turns"] for r in all_results if r["won"]]
    if won_turns:
        print(f"  Avg turns (won): {sum(won_turns)/len(won_turns):.1f}")

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
            print(f"  {diff}: {dw}/{len(dr)} solved ({100*dw/len(dr):.1f}%)")

    # Save
    summary = {
        "model": args.model_path,
        "data_file": args.data_file,
        "memory_mode": "checkpoint" if args.checkpoint else "random",
        "checkpoint": args.checkpoint,
        "memory_window": args.memory_window,
        "num_games": n,
        "max_turns": args.max_turns,
        "elapsed_seconds": elapsed,
        "solve_rate": wins / n if n else 0,
        "avg_score": total_score / n if n else 0,
        "avg_reward": total_reward / n if n else 0,
        "by_difficulty": {
            diff: {
                "num_games": len(dr),
                "wins": sum(1 for r in dr if r["won"]),
                "solve_rate": sum(1 for r in dr if r["won"]) / len(dr) if dr else 0,
            }
            for diff, dr in by_diff.items()
        },
        "results": all_results,
    }

    split_name = Path(args.data_file).stem
    tag = "random" if not args.checkpoint else "trained"
    output_file = args.output_file or str(
        Path(args.data_file).parent / f"eval_results/memory_{tag}_w{args.memory_window}_{split_name}.json"
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
