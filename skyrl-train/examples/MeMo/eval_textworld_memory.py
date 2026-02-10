#!/usr/bin/env python3
"""
Evaluate TextWorld with Memory module using SentenceTransformer encoding.

Uses the same encoding pipeline as RL training:
  1. SentenceTransformer (Qwen3-Embedding-4B) embeds memory documents
  2. Memory module cross-attention produces 8 memory token embeddings
  3. Embeddings replace <|image_pad|> placeholders via vLLM EmbedsPrompt

Usage:
    # With trained checkpoint (match window to training)
    python eval_textworld_memory.py --data_file ~/data/textworld_memo/test.parquet \
        --checkpoint /path/to/memory_sft.pt --memory_window 5

    # Random (untrained) memory
    python eval_textworld_memory.py --data_file ~/data/textworld_memo/test.parquet
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
from vllm.inputs import EmbedsPrompt


_TW_LOCK = threading.Lock()
PLACEHOLDER_TOKEN = "<|image_pad|>"
NUM_MEMORIES = 8


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
    current_segment: List[Dict] = field(default_factory=list)
    memory_documents: List[str] = field(default_factory=list)
    max_memory_docs: int = 20


def load_memory_and_encoder(
    memo_repo_root: str,
    checkpoint: Optional[str],
    embedding_model_name: str,
    embedding_device: str = "cuda",
):
    """Load Memory module + SentenceTransformer encoder."""
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
        missing, unexpected = memory.load_state_dict(state_dict, strict=False)
        loaded = len(list(memory.state_dict().keys())) - len(missing)
        print(f"Loaded memory checkpoint: {checkpoint} ({loaded} params, {len(missing)} missing)")
    else:
        print("Using RANDOM (untrained) memory module")

    memory.eval()
    memory.cuda()

    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model {embedding_model_name} on {embedding_device}...")
    embed_model = SentenceTransformer(
        embedding_model_name,
        model_kwargs={"device_map": embedding_device, "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    print("Embedding model loaded.")

    return memory, embed_model


def encode_documents_st(
    memory_module,
    embed_model,
    documents: List[str],
    embedding_device: str = "cuda",
) -> Optional[torch.Tensor]:
    """Encode memory documents using SentenceTransformer + Memory module.

    Matches the training pipeline (MemoSentenceEmbeddingEncoder.encode).
    All docs bundled as [1, num_docs, 2560] for cross-attention.

    Returns: (NUM_MEMORIES, 2560) tensor on CUDA, or None if no docs.
    """
    if not documents:
        return None

    with torch.no_grad():
        embeddings = embed_model.encode(
            documents,
            convert_to_numpy=False,
            show_progress_bar=False,
            device=embedding_device,
        )

    if isinstance(embeddings, torch.Tensor):
        doc_embeds = embeddings
    else:
        doc_embeds = torch.stack([
            t if isinstance(t, torch.Tensor) else torch.tensor(t)
            for t in embeddings
        ])

    # [num_docs, 2560] -> [1, num_docs, 2560]
    mem_dtype = next(memory_module.parameters()).dtype
    doc_embeds = doc_embeds.unsqueeze(0).to(device="cuda", dtype=mem_dtype)

    with torch.no_grad():
        memory_embeddings, _ = memory_module(doc_embeds, padding_mask=None)

    # [1, NUM_MEMORIES, 2560] -> [NUM_MEMORIES, 2560]
    return memory_embeddings.squeeze(0)


def build_prompt_with_memory_embeds(
    messages: List[Dict[str, str]],
    memory_embeds: Optional[torch.Tensor],
    tokenizer,
    llm_embed_weight: torch.Tensor,
    placeholder_id: int,
    max_length: int = 8000,
) -> EmbedsPrompt:
    """Build an EmbedsPrompt with memory embeddings replacing placeholder tokens."""
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    # Truncate from the left (keep recent context) if too long
    if len(token_ids) > max_length:
        token_ids = token_ids[-max_length:]
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=llm_embed_weight.device)
    prompt_embeds = torch.nn.functional.embedding(token_tensor, llm_embed_weight)  # [seq_len, dim]

    if memory_embeds is not None:
        memory_embeds = memory_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        placeholder_positions = (token_tensor == placeholder_id).nonzero(as_tuple=True)[0]
        num_to_replace = min(len(placeholder_positions), memory_embeds.shape[0])
        for i in range(num_to_replace):
            prompt_embeds[placeholder_positions[i]] = memory_embeds[i]

    return EmbedsPrompt(prompt_embeds=prompt_embeds.unsqueeze(0))


def main():
    parser = argparse.ArgumentParser(description="TextWorld evaluation with Memory module")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--memo_repo_root", type=str, default=str(Path.home() / "MeMo"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Memory module checkpoint (None=random)")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding_device", type=str, default="cuda")
    parser.add_argument("--memory_window", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_memory_docs", type=int, default=20)
    parser.add_argument("--max_generate_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory", type=float, default=0.50)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
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

    # Load dataset (deduplicate by game_file)
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
    print(f"Embedding: {args.embedding_model} on {args.embedding_device}")
    print(f"Window: {args.memory_window}, Max docs: {args.max_memory_docs}")
    print(f"Games: {len(eval_rows)}, GPU mem util: {args.gpu_memory}")

    # Load memory module + sentence transformer
    memory_module, embed_model = load_memory_and_encoder(
        args.memo_repo_root, args.checkpoint, args.embedding_model, args.embedding_device
    )

    # Load tokenizer and LLM embedding weight for prompt construction
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    from safetensors.torch import load_file as load_safetensors
    model_dir = Path(args.model_path)
    llm_embed_weight = None
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        tensors = load_safetensors(str(sf_file))
        for name in ("model.embed_tokens.weight", "model.wte.weight"):
            if name in tensors:
                llm_embed_weight = tensors[name].cuda().to(torch.bfloat16)
                break
        if llm_embed_weight is not None:
            break
    if llm_embed_weight is None:
        raise RuntimeError("Could not find embedding weight in model files")
    print(f"LLM embedding weight: {llm_embed_weight.shape}")

    # Load vLLM
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        seed=args.seed,
        trust_remote_code=True,
        enable_prompt_embeds=True,
    )

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

    print(f"\nEvaluating {total_games} games (window={args.memory_window}, max_turns={args.max_turns})", flush=True)
    print("=" * 70, flush=True)

    for batch_start in range(0, total_games, args.batch_size):
        batch_rows = eval_rows[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total_games + args.batch_size - 1) // args.batch_size
        print(f"\nBatch {batch_num}/{total_batches}: games {batch_start+1}-{batch_start+len(batch_rows)}", flush=True)

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
        print(f"  Games initialized, starting turns...", flush=True)

        # Run turns
        for turn in range(1, args.max_turns + 1):
            active = [g for g in games if not g.done]
            if not active:
                break

            prompts = []
            for g in active:
                obs = g.tw_state.feedback.strip()

                g.current_segment.append({
                    "turn": turn,
                    "observation": obs,
                    "action": "",
                    "reward": 0.0,
                    "score": getattr(g.tw_state, "score", 0),
                })

                # Update system message with placeholder block if we have memory docs
                if g.memory_documents:
                    placeholder_block = " ".join([PLACEHOLDER_TOKEN] * NUM_MEMORIES)
                    g.messages[0]["content"] = g.system_prompt + "\n\nMemory context:\n" + placeholder_block

                g.messages.append({"role": "user", "content": obs})

                # Encode memory documents and build EmbedsPrompt
                memory_embeds = encode_documents_st(
                    memory_module, embed_model, g.memory_documents, args.embedding_device
                )
                prompt = build_prompt_with_memory_embeds(
                    g.messages, memory_embeds, tokenizer, llm_embed_weight, placeholder_id
                )
                prompts.append(prompt)

            # Batched generation with actual memory embeddings injected
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for g, output in zip(active, outputs):
                raw_response = output.outputs[0].text
                action = parse_action(raw_response)

                g.actions.append(action)
                g.messages.append({"role": "assistant", "content": raw_response})
                g.num_turns = len(g.actions)
                g.current_segment[-1]["action"] = action

                g.tw_state, reward, done = g.env.step(action)
                g.total_reward += float(reward)
                g.final_score = getattr(g.tw_state, "score", 0)
                g.won = bool(done and getattr(g.tw_state, "won", False))
                g.done = done or turn >= args.max_turns

                g.current_segment[-1]["reward"] = float(reward)
                g.current_segment[-1]["score"] = g.final_score

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

        # Running tally
        wins_so_far = sum(1 for r in all_results if r["won"])
        print(f"  -- Running: {wins_so_far}/{len(all_results)} ({100*wins_so_far/len(all_results):.1f}%)")

    elapsed = time.time() - t_start
    n = len(all_results)
    wins = sum(1 for r in all_results if r["won"])

    print(f"\n{'=' * 70}")
    mode = f"checkpoint={args.checkpoint}" if args.checkpoint else "RANDOM"
    print(f"MEMORY RESULTS ({mode}, window={args.memory_window}, {n} games, {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Solve rate: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Avg score:  {sum(r['final_score'] for r in all_results)/n:.2f}")
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
        "embedding_model": args.embedding_model,
        "memory_window": args.memory_window,
        "num_games": n,
        "max_turns": args.max_turns,
        "elapsed_seconds": elapsed,
        "solve_rate": wins / n if n else 0,
        "wins": wins,
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
