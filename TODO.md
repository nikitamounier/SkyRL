# MeMo TextWorld — TODO & Analysis (Updated Feb 9, 2026)

---

## Deep Analysis of All Runs (Feb 9)

### Changes Made So Far
1. **Embedding model**: Swapped from LLM's `embed_tokens` to `Qwen3-Embedding-4B` via SentenceTransformer (semantic vs raw token lookup)
2. **Multi-doc bundling**: Fixed [1,1,2560] → [1,num_docs,2560] so cross-attention has multiple positions
3. **Zero-init → Small-init**: Pure zeros made memory invisible (grad=nan, no learning). Changed to `std=0.01`
4. **Reward shaping**: `step_penalty=0.01`, `efficiency_bonus=1.0`
5. **Speed**: `max_model_len=4224` (was 262K), `max_prompt_length=4096`, `max_generate_length=128`
6. **System prompts**: Updated with valid TextWorld commands and strategy tips
7. **OpenAI SFT script**: `play_textworld_openai.py` for generating expert gameplay data

### Eval Baselines (no memory, no RL)

| Model | Prompt | Test | Validation |
|-------|--------|------|------------|
| Qwen3-4B | Old (minimal) | 4/20 (20%) | 16/40 (40%) |
| Qwen3-4B | New (commands+strategy) | 5/20 (25%) | 20/40 (50%) |
| **GPT-5-mini** | **New** | **15/20 (75%)** | — |

Key insight: Qwen3-4B spams `[ACTION: move south]` (wrong syntax) 50 times. GPT-5-mini reasons and solves games in 8-14 turns. The base model barely plays the game — memory can't help a model that doesn't know the commands.

### Training Run Results (Small-init, Feb 9, ~20 runs)

**Best runs** (reward improving, stable entropy <0.5):
| Run | Steps | First3 R | Last3 R | Trend | Grad | Entropy |
|-----|-------|----------|---------|-------|------|---------|
| grpo_w3 | 16 | 0.16 | **0.58** | UP | 0.02 | 0.27 |
| grpo_w5 | 19 | 0.17 | 0.23 | flat | 10 | 0.35 |
| grpo_w5_lr5e4 | 10 | 0.16 | **0.33** | UP | 3 | 0.23 |
| reinforce_w10 | 79 | 0.13 | **1.44** | UP | 0.06 | 0.20 |
| reinforce_w5_t100 | 57 | 0.24 | 0.13 | flat | 0.6 | 0.41 |

**Exploded runs** (entropy >1, grad >1000):
| Run | Entropy | Grad | What happened |
|-----|---------|------|---------------|
| reinforce_w3 | 4.1 | 98M | Fully exploded |
| grpo_w10 | 8.4 | 1530 | Fully exploded |
| reinforce_w5_lr5e4 | 4.9 | 7070 | LR too high |
| reinforce_w5_lr1e5 | 0.26 | 3255 | Grad spikes |

**Flat runs** (no improvement, not exploding):
| Run | Steps | First3 R | Last3 R | Note |
|-----|-------|----------|---------|------|
| grpo_w5_cosine | 63 | 0.06 | 0.06 | Cosine decay too aggressive |
| grpo_w5_lr5e5 | 32 | 0.11 | 0.11 | LR too low |
| grpo_w5_t100 | 56 | 0.19 | 0.17 | 100 turns too long |
| reinforce_w5_n8 | 40 | 0.05 | -0.08 | Getting worse |

### Key Conclusions

1. **Small-init (std=0.01) works** — gradients flow, clip_ratio > 0. This was the critical fix after zero-init produced nan gradients for 9 hours.

2. **GRPO w3 is the best config** — reward 0.16 → 0.58 (only 16 steps). Smaller window = fewer docs = less chance of gradient explosion. The cross-attention is more stable with 1-2 docs than 5-10.

3. **REINFORCE++ w10 is a slow burner** — 79 steps, reward 0.13 → 1.44. Stable but slow convergence. The most data we have on any run.

4. **w10 + GRPO explodes, w3 + REINFORCE++ explodes** — there's a pattern: larger windows need REINFORCE++ (more stable), smaller windows can use GRPO (faster). The mismatch causes instability.

5. **Higher LR (5e-4) is too aggressive** — causes entropy explosion. LR=1e-4 is the sweet spot. LR=5e-5 is too slow.

6. **n_samples=8 doesn't help much** — no clear improvement over n_samples=5. The extra samples slow things down.

7. **t100 (100 turns) doesn't help** — same reward as t50 but 2x slower per step.

8. **The model can't really play TextWorld** — it outputs `move south` instead of `go south`. Even with memory, a model that doesn't know the game commands won't improve.

9. **GPT-5-mini solves 75% of games** — the games are solvable. The gap is model capability, not memory.

### Root Cause Analysis (Updated)

The RL training is fundamentally limited by:
1. **Base model can't play the game** — spams wrong commands, doesn't follow objectives
2. **Memory can't fix a broken policy** — if the model doesn't know `go north`, memory of past rooms is useless
3. **Reward signal is too sparse** — even with step penalty, most games score 0
4. **Gradient instability** — memory outputs small perturbations that periodically destabilize the LLM

### What Would Actually Move The Needle

**Priority 1: SFT on GPT-5-mini transcripts**
- We have 15/20 test games solved by GPT-5-mini
- Generate transcripts for all 400 train games (~$50-100 in API costs)
- SFT Qwen3-4B on these transcripts (teach it how to play TextWorld)
- This would jump baseline from 25% to maybe 50-60%

**Priority 2: RL on top of SFT model**
- Once the model can actually play, memory becomes meaningful
- The RL + memory training can then improve on the SFT baseline
- Use grpo_w3 or reinforce_w10 configs (proven stable)

**Priority 3: New prompts in training**
- Dataset v3 with proper TextWorld commands is ready at `/large_storage/goodarzilab/parsaidp/MeMo/data/textworld_memo_v3/`
- Need to point training runs at this data
- Will help the model learn correct command syntax

---

## Active TODO
- [x] Deep analysis of all runs — see above
- [ ] Generate GPT-5-mini transcripts for full train set (400 games) — **HIGH PRIORITY**
- [ ] SFT Qwen3-4B on GPT-5-mini transcripts
- [ ] Rerun RL training with SFT model as base + new prompts (v3 dataset)
- [ ] Fix eval script to properly inject memory embeddings
- [ ] Try starting from MeMo personalization checkpoint

## Running Experiments
- 19 training runs still cooking (small-init, various configs)
- GPT-5-mini test set DONE: 15/20 (75%)
- Dataset v3 (new prompts) DONE: `/large_storage/goodarzilab/parsaidp/MeMo/data/textworld_memo_v3/`

## Done
- [x] Swap embed_tokens to Qwen3-Embedding-4B
- [x] Fix multi-doc bundling [1,1,2560] → [1,num_docs,2560]
- [x] Zero-init → Small-init (std=0.01) for memory_projection
- [x] Add reward shaping (step_penalty=0.01, efficiency_bonus=1.0)
- [x] Add REINFORCE++ as alternative to GRPO
- [x] Fix max_model_len (262K → 4224)
- [x] Lower max_prompt_length (16384 → 4096) and max_generate_length (256 → 128)
- [x] Bump slurm memory 64G → 96G
- [x] Make configs overridable via env vars
- [x] Run baseline evals on all splits (old + new prompts)
- [x] GPT-5-mini eval on test set (75% solve rate)
- [x] Create OpenAI SFT data generation script
- [x] Update system prompts with valid TextWorld commands
- [x] Generate dataset v3 with new prompts

## All Eval Results

| Model | Prompt | Test | Validation | Train |
|-------|--------|------|------------|-------|
| Qwen3-4B | Old | 4/20 (20%) | 16/40 (40%) | 90/340 (26.5%) |
| Qwen3-4B | New | 5/20 (25%) | 20/40 (50%) | — |
| GPT-5-mini | New | 15/20 (75%) | — | — |

## Key Files
- Analysis doc: `docs/memo_training_analysis_2026-02-08.md`
- Training script: `skyrl-train/examples/MeMo/run_textworld_memo_train.sh`
- Slurm: `train_textworld_memo.slurm`
- Launcher: `launch_comparison.sh`
- Encoder: `skyrl-train/skyrl_train/examples/modalities/memo_handlers.py`
- Env: `skyrl-gym/skyrl_gym/envs/textworld/env.py`
- Data (old prompts): `/home/parsaidp/data/textworld_memo/`
- Data (new prompts): `/large_storage/goodarzilab/parsaidp/MeMo/data/textworld_memo_v3/`
- Games: `/home/parsaidp/data/textworld_memo/games_v2/`
- GPT-5 transcripts: `/large_storage/goodarzilab/parsaidp/MeMo/data/sft_gpt5mini_test/`
- OpenAI script: `skyrl-train/examples/MeMo/play_textworld_openai.py`
- Eval script (new prompt): `skyrl-train/examples/MeMo/eval_textworld_newprompt.py`
- MeMo reference: `/home/parsaidp/MeMo/`
