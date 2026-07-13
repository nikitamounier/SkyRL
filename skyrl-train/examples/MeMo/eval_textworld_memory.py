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

# For .json game files, the textworld lib only returns stub feedback
# ("[To get text observation use the '.z8' file instead of the '.json' one.]"),
# so wrap FastTextWorldSimulator with a tw_state-compatible shim.
from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator


class _TwState:
    """Mimics textworld GameState attrs the eval reads (feedback, score, won)."""
    __slots__ = ("feedback", "score", "won")
    def __init__(self, feedback: str, score: int, won: bool):
        self.feedback = feedback
        self.score = score
        self.won = won


class _FastEnvAdapter:
    """Adapter so FastTextWorldSimulator quacks like a textworld env."""
    def __init__(self, json_path: str):
        self._sim = FastTextWorldSimulator(json_path)

    def reset(self):
        obs, info = self._sim.reset()
        return _TwState(feedback=obs, score=int(info.get("score") or 0), won=bool(info.get("won") or False))

    def step(self, action: str):
        obs, reward, done, info = self._sim.step(action)
        st = _TwState(feedback=obs, score=int(info.get("score") or 0), won=bool(info.get("won") or False))
        return st, float(reward), bool(done)

    def close(self):
        pass


def _make_env(game_file: str):
    """Pick the right backend: textworld lib for .z8, FastTextWorldSimulator for .json."""
    if game_file.endswith(".json"):
        return _FastEnvAdapter(game_file)
    return textworld.start(game_file)
from vllm.inputs import EmbedsPrompt


_TW_LOCK = threading.Lock()
PLACEHOLDER_TOKEN = "<|image_pad|>"
NUM_MEMORIES = 8

# Inform7/TextWorld command vocabulary hint — mirror of fast_env.py
# _TEXTWORLD_VOCAB_HINT so eval and training share the same system content.
_TEXTWORLD_VOCAB_HINT_EVAL = """

Valid TextWorld commands (use exactly these forms — variations are rejected by the parser):
- `look` (describe room — NOT `look around` or `look at <room>`)
- `inventory` (list held items)
- `examine <object>` (NOT `examine <room>` or `inspect`)
- `take <object>` (NOT `pick up`, `grab`, `get`)
- `drop <object>`
- `open <object>` / `close <object>` (for doors and containers)
- `unlock <object> with <key>`
- `put <object> in <container>` / `put <object> on <surface>`
- `go <direction>` or just `<direction>` (north/south/east/west/up/down) — NOT `move`, `walk`, `head`
If a command is rejected (`You can't see any such thing.` or similar), DO NOT repeat it — try a different valid form."""


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
    seed_user: Optional[str] = None  # seed user message before first obs (from dump)
    env: Any = None
    tw_state: Any = None
    last_observation: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    total_reward: float = 0.0
    last_reward: float = 0.0
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
    from memo.models.memory import AdaptiveMemory

    # Use AdaptiveMemory to match the pool_mode='adaptive' RL training config.
    # Args mirror what run_textworld_memo_train.sh passes via Hydra:
    #   num_heads=8, num_self_attn_layers=1, num_cross_attn_layers=2, use_gate=False.
    memory = AdaptiveMemory(
        embedding_dim=2560,
        num_memories=NUM_MEMORIES,
        output_dim=2560,
        num_heads=8,
        num_self_attn_layers=1,
        num_cross_attn_layers=2,
        dropout=0.1,
        memory_init="xavier_uniform",
        projection_type="linear",
        # use_gate MUST match training. The RL training factory (create_memory in
        # memo_handlers.py) does NOT pass use_gate, so AdaptiveMemory defaults to
        # use_gate=False. Setting True here would add untrained slot_gate.* params
        # (missing keys) and apply a sigmoid gate the policy never saw -> garbage.
        use_gate=False,
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
    # CRITICAL dtype match: in training the memory module runs in its native
    # fp32 (the modality encoder lives outside FSDP and is never downcast;
    # MemoSentenceEmbeddingEncoder.encode casts inputs to the module's fp32 param
    # dtype and runs the forward in fp32). Only the FINAL memory vectors are cast
    # to the policy/vLLM dtype (bf16) at injection time (vllm_engine
    # .compute_embeddings -> target_dtype = bf16). Running the whole module in
    # bf16 produces numerically different vectors and collapses greedy decoding
    # for some checkpoints (step20/25). So keep the module fp32 here and cast only
    # the produced vectors to bf16 inside build_prompt_with_memory_embeds.
    memory.cuda().float()

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
    observation: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Encode memory documents using SentenceTransformer + Memory module.

    Matches the training pipeline (MemoSentenceEmbeddingEncoder.encode):
    - docs are batched into [1, D, 2560]
    - observation (if provided) is embedded and used as question_embeds
    - else falls back to mean-pooled doc embeddings

    Returns: (NUM_MEMORIES, 2560) tensor on CUDA. When there are no documents,
    returns zeros — matching MemoSentenceEmbeddingEncoder.encode, which returns
    zero memory vectors for empty payloads (so the prepended placeholder slots
    get zero embeddings rather than the raw pad-token embedding).
    """
    mem_dtype = next(memory_module.parameters()).dtype
    out_dim = getattr(memory_module, "output_dim", 2560)
    if not documents:
        return torch.zeros(NUM_MEMORIES, out_dim, device="cuda", dtype=mem_dtype)

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
    doc_embeds = doc_embeds.unsqueeze(0).to(device="cuda", dtype=mem_dtype)

    # AdaptiveMemory needs a question_embeds signal. Match training: embed the
    # current observation and use it as the question. Fall back to mean-pooled
    # docs if no obs provided (matches MemoSentenceEmbeddingEncoder fallback).
    obs_str = observation.strip() if isinstance(observation, str) else ""
    if obs_str:
        with torch.no_grad():
            obs_raw = embed_model.encode(
                [obs_str],
                convert_to_numpy=False,
                show_progress_bar=False,
                device=embedding_device,
            )
        if isinstance(obs_raw, torch.Tensor):
            q_e = obs_raw
        else:
            q_e = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in obs_raw])
        question_embeds = q_e.to(device="cuda", dtype=mem_dtype).unsqueeze(1)  # [1, 1, 2560]
    else:
        question_embeds = doc_embeds.mean(dim=1, keepdim=True)  # [1, 1, 2560]

    with torch.no_grad():
        out = memory_module(
            inputs_embeds=doc_embeds,
            question_embeds=question_embeds,
            doc_padding_mask=None,
        )
    memory_embeddings = out[0] if isinstance(out, (tuple, list)) else out

    # [1, NUM_MEMORIES, 2560] -> [NUM_MEMORIES, 2560]
    return memory_embeddings.squeeze(0)


def build_prompt_token_ids(messages: List[Dict[str, str]], tokenizer) -> List[int]:
    """Token ids = (NUM_MEMORIES placeholder tokens) PREPENDED, then the chat
    template of `messages` with a generation prompt.

    This mirrors the in-training eval input EXACTLY (verified against the dumped
    eval input_prompt): the 8 `<|image_pad|>` placeholders sit at the very start,
    BEFORE `<|im_start|>system`, followed by the standard chat-templated history.
    """
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_ids = tokenizer.encode(chat_str, add_special_tokens=False)
    return [placeholder_id] * NUM_MEMORIES + chat_ids


def build_prompt_with_memory_embeds(
    messages: List[Dict[str, str]],
    memory_embeds: Optional[torch.Tensor],
    tokenizer,
    llm_embed_weight: torch.Tensor,
    placeholder_id: int,
    max_length: int = 8000,
) -> EmbedsPrompt:
    """Build an EmbedsPrompt with memory embeddings replacing placeholder tokens.

    Placeholders are PREPENDED (see build_prompt_token_ids) to match training.
    """
    token_ids = build_prompt_token_ids(messages, tokenizer)
    # Truncate from the left (keep recent context) if too long. The 8 prepended
    # placeholders live at the very start; left-truncation would drop them, so we
    # only ever hit this path for prompts we already decided to keep (caller
    # ends the episode when over budget), making this a safety no-op.
    if len(token_ids) > max_length:
        token_ids = token_ids[:NUM_MEMORIES] + token_ids[NUM_MEMORIES:][-(max_length - NUM_MEMORIES):]
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=llm_embed_weight.device)
    prompt_embeds = torch.nn.functional.embedding(token_tensor, llm_embed_weight)  # [seq_len, dim]

    if memory_embeds is not None:
        # Inject in the SAME dtype as the LLM embedding space (bf16) — matches the
        # training/in-training modality path, which runs the memory module and
        # injects prompt_embeds in the policy dtype (bf16). An fp32-vs-bf16
        # difference in the injected vectors can flip greedy token choices.
        memory_embeds = memory_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        placeholder_positions = (token_tensor == placeholder_id).nonzero(as_tuple=True)[0]
        num_to_replace = min(len(placeholder_positions), memory_embeds.shape[0])
        for i in range(num_to_replace):
            prompt_embeds[placeholder_positions[i]] = memory_embeds[i]

    return EmbedsPrompt(prompt_embeds=prompt_embeds.unsqueeze(0))


def main():
    parser = argparse.ArgumentParser(description="TextWorld evaluation with Memory module")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Parquet eval set. Optional when --prompts_from_dump is given.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--memo_repo_root", type=str, default=str(Path.home() / "MeMo"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Memory module checkpoint (None=random)")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding_device", type=str, default="cuda")
    # Defaults match the sweep1c TextWorldEnv (env.py) RL training/eval config:
    #   memory_window=1, max_memory_docs=60, eval temp=0.0/top_p=1.0,
    #   max_generate_length=128, max_model_len=4224, max_prompt=4096.
    parser.add_argument("--memory_window", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--max_memory_docs", type=int, default=60)
    parser.add_argument("--max_generate_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gpu_memory", type=float, default=0.50)
    parser.add_argument("--max_model_len", type=int, default=4224)
    parser.add_argument("--max_prompt_length", type=int, default=4096,
                        help="Episode ends (stop_reason=length) when prompt exceeds this, matching training max_input_length.")
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompts_from_dump", type=str, default=None,
        help="JSON list of {game_file, system, seed_user} reconstructed from the in-training "
             "dumped_evals. When set, the per-game system prompt + seed user message exactly "
             "match what the checkpoint was evaluated on (overrides --data_file prompts).",
    )
    args = parser.parse_args()

    # Resolve local snapshot
    if args.model_path == "Qwen/Qwen3-4B-Instruct-2507":
        cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
        if cache_dir.exists():
            snapshots = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                args.model_path = str(snapshots[0])

    # Load eval rows. Prefer the dump-reconstructed per-game prompts (exact match
    # to the in-training eval); otherwise fall back to the parquet `prompt` field.
    if args.prompts_from_dump:
        recs = json.load(open(args.prompts_from_dump))
        eval_rows = [
            {
                "game_file": r["game_file"],
                # synthesize a 2-message prompt: [system, seed_user] exactly as the
                # in-training eval fed it (verified vs dumped input_prompt).
                "prompt": [
                    {"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["seed_user"]},
                ],
            }
            for r in recs
        ]
        seen = {r["game_file"] for r in eval_rows}
    else:
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
            # Seed user message (the message before the first observation). With
            # --prompts_from_dump this is the EXACT per-game seed the in-training
            # eval used (e.g. "Continue the game."). Falls back to the parquet's
            # second message if present.
            seed_user = None
            if len(prompt_msgs) > 1 and prompt_msgs[1].get("role") == "user":
                seed_user = prompt_msgs[1]["content"]
            gs = GameState(
                idx=batch_start + i,
                game_file=row["game_file"],
                system_prompt=system_prompt,
                seed_user=seed_user,
                memory_window=args.memory_window,
                messages=[{"role": "system", "content": system_prompt}],
                max_memory_docs=args.max_memory_docs,
            )
            with _TW_LOCK:
                gs.env = _make_env(row["game_file"])
            gs.tw_state = gs.env.reset()
            games.append(gs)
        print(f"  Games initialized, starting turns...", flush=True)

        # Seed the conversation to match the in-training eval EXACTLY:
        #   [system, seed_user, first_obs]  (then chat history ACCUMULATES).
        # The 8 memory placeholders are prepended at the very start at prompt-build
        # time (build_prompt_token_ids), before <|im_start|>system.
        for g in games:
            obs0 = g.tw_state.feedback.strip()
            msgs = [{"role": "system", "content": g.system_prompt}]
            if g.seed_user:
                msgs.append({"role": "user", "content": g.seed_user})
            msgs.append({"role": "user", "content": obs0})
            g.messages = msgs
            g.last_observation = obs0

        # Run turns — replicate the in-training eval (verified vs dumped_evals):
        #   * 8 memory placeholders PREPENDED at the prompt prefix every turn
        #     (from dataset preprocessing), regardless of whether docs exist.
        #   * memory content refreshed each step from current docs; encoder returns
        #     zero vectors when there are no docs yet (matches MemoSentenceEmbedding
        #     Encoder.encode -> runtime placeholder replacement with zeros).
        #   * AdaptiveMemory question signal = current observation (env.py).
        #   * full accumulating chat history (system, seed_user, obs, asst, obs...).
        #   * memory module + injected vectors run in bf16 (policy dtype).
        for turn in range(1, args.max_turns + 1):
            active = [g for g in games if not g.done]
            if not active:
                break

            prompts = []
            gen_games = []
            for g in active:
                obs = g.last_observation

                # Episode ends if the prompt (incl. 8 prepended placeholders)
                # exceeds the training max_input_length budget.
                token_ids = build_prompt_token_ids(g.messages, tokenizer)
                if len(token_ids) > args.max_prompt_length:
                    g.done = True
                    continue

                # Always produce 8 memory vectors. When no finalized docs exist,
                # encode_documents_st returns zeros (num_memories x dim).
                memory_embeds = encode_documents_st(
                    memory_module, embed_model, g.memory_documents, args.embedding_device,
                    observation=obs,
                )

                prompt = build_prompt_with_memory_embeds(
                    g.messages, memory_embeds, tokenizer, llm_embed_weight, placeholder_id,
                    max_length=args.max_prompt_length,
                )
                prompts.append(prompt)
                gen_games.append(g)

            if not prompts:
                continue

            # Batched generation with actual memory embeddings injected
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for g, output in zip(gen_games, outputs):
                raw_response = output.outputs[0].text
                action = parse_action(raw_response)

                g.actions.append(action)
                g.num_turns = len(g.actions)

                # Append assistant turn (full raw response) — accumulating history.
                g.messages.append({"role": "assistant", "content": raw_response})

                # Step the env. The memory doc for this turn pairs the observation
                # the model SAW (g.last_observation) with the action it took and
                # the NEW score — exactly _IncrementalMemory.add_turn in env.py.
                prev_obs = g.last_observation
                g.tw_state, reward, done = g.env.step(action)
                g.total_reward += float(reward)
                g.final_score = getattr(g.tw_state, "score", 0)
                g.last_reward = float(reward)  # in-training pass@1 = (last-step reward > 0)
                g.won = bool(done and getattr(g.tw_state, "won", False))
                g.done = done or turn >= args.max_turns

                g.current_segment.append({
                    "turn": turn,
                    "observation": prev_obs[:200],   # env.py truncates obs to [:200]
                    "action": action,
                    "raw_response": raw_response,
                    "reward": float(reward),
                    "score": g.final_score,
                })

                # Finalize memory window (window=1 -> every turn becomes a doc).
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

                # Append the new observation as the next user message (unless done).
                new_obs = g.tw_state.feedback.strip()
                g.last_observation = new_obs
                if not g.done:
                    g.messages.append({"role": "user", "content": new_obs})

        # Collect results
        for g in games:
            g.env.close()
            # `passed` matches the in-training pass@1 metric EXACTLY: a trajectory
            # passes iff its LAST-step reward is > 0 (get_metrics_from_generator_output
            # uses rewards[-1] > 0), not whether the env's `won` flag is set.
            passed = g.last_reward > 0
            result = {
                "game_file": g.game_file,
                "passed": passed,
                "won": g.won,
                "final_score": g.final_score,
                "total_reward": g.total_reward,
                "last_reward": g.last_reward,
                "num_turns": g.num_turns,
                "num_memory_docs": len(g.memory_documents),
            }
            all_results.append(result)
            status = "PASS" if passed else f"score={g.final_score}"
            print(f"  [{g.idx+1}/{total_games}] {Path(g.game_file).stem}: {status} ({g.num_turns} turns, {len(g.memory_documents)} docs)")

        # Running tally (pass@1)
        wins_so_far = sum(1 for r in all_results if r["passed"])
        print(f"  -- Running: {wins_so_far}/{len(all_results)} ({100*wins_so_far/len(all_results):.1f}%)")

    elapsed = time.time() - t_start
    n = len(all_results)
    wins = sum(1 for r in all_results if r["passed"])

    print(f"\n{'=' * 70}")
    mode = f"checkpoint={args.checkpoint}" if args.checkpoint else "RANDOM"
    print(f"MEMORY RESULTS ({mode}, window={args.memory_window}, {n} games, {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Solve rate: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Avg score:  {sum(r['final_score'] for r in all_results)/n:.2f}")
    print(f"  Avg turns:  {sum(r['num_turns'] for r in all_results)/n:.1f}")
    won_turns = [r["num_turns"] for r in all_results if r["passed"]]
    if won_turns:
        print(f"  Avg turns (won): {sum(won_turns)/len(won_turns):.1f}")

    # Tier is the token right after the game id in the stem, e.g.
    #   game_00122_hard_w27_o33_q11_s1226721 -> "hard"
    #   game_00098_xlong_w54_o118_q68_s609023 -> "xlong"
    # Known tiers, ordered easy->hard. "big-tier" = long/xlong/huge/mega(+ any
    # ICL-doc aliases like mega_ultra/extreme), where ICL scores 0%.
    KNOWN_TIERS = [
        "tiny", "small", "easy", "medium", "hard",
        "long", "xlong", "huge", "mega", "mega_ultra", "ultra", "extreme",
    ]
    BIG_TIERS = {"long", "xlong", "huge", "mega", "mega_ultra", "ultra", "extreme"}

    def _tier_of(stem: str) -> str:
        parts = stem.split("_")
        # stem starts with game_<id>_<tier>_...; the tier is the first known
        # tier token found in the stem.
        for p in parts:
            if p in KNOWN_TIERS:
                return p
        return "unknown"

    by_diff: Dict[str, list] = {}
    for r in all_results:
        diff = _tier_of(Path(r["game_file"]).stem)
        by_diff.setdefault(diff, []).append(r)
        r["tier"] = diff

    if by_diff:
        print(f"\nBy tier:")
        order = KNOWN_TIERS + ["unknown"]
        for diff in order:
            if diff not in by_diff:
                continue
            dr = by_diff[diff]
            dw = sum(1 for r in dr if r["passed"])
            print(f"  {diff}: {dw}/{len(dr)} solved ({100*dw/len(dr):.1f}%)")

    # big-tier rollup (the long-game band where ICL = 0%)
    big = [r for r in all_results if r.get("tier") in BIG_TIERS]
    if big:
        bw = sum(1 for r in big if r["passed"])
        print(f"  big-tier (long/xlong/huge/mega): {bw}/{len(big)} solved ({100*bw/len(big):.1f}%)")

    by_tier_summary = {
        t: {
            "solved": sum(1 for r in rs if r["passed"]),
            "total": len(rs),
            "solve_rate": (sum(1 for r in rs if r["passed"]) / len(rs)) if rs else 0,
        }
        for t, rs in by_diff.items()
    }

    # Save
    summary = {
        "model": args.model_path,
        "data_file": args.data_file or args.prompts_from_dump,
        "memory_mode": "checkpoint" if args.checkpoint else "random",
        "checkpoint": args.checkpoint,
        "embedding_model": args.embedding_model,
        "memory_window": args.memory_window,
        "num_games": n,
        "max_turns": args.max_turns,
        "elapsed_seconds": elapsed,
        "solve_rate": wins / n if n else 0,
        "wins": wins,
        "by_tier": by_tier_summary,
        "results": all_results,
    }

    source_path = args.data_file or args.prompts_from_dump or "eval"
    split_name = Path(source_path).stem
    tag = "random" if not args.checkpoint else "trained"
    output_file = args.output_file or str(
        Path(source_path).parent / f"eval_results/memory_{tag}_w{args.memory_window}_{split_name}.json"
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
