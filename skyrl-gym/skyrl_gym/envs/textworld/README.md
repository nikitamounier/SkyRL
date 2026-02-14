# TextWorld Environments

Two TextWorld environment implementations for SkyRL RL training.

## Overview

| | `TextWorldEnv` (original) | `FastTextWorldEnv` (new) |
|---|---|---|
| **File** | `env.py` | `fast_env.py` + `fast_sim.py` |
| **Engine** | Inform7 Z-machine subprocess | Pure-Python JSON interpreter |
| **Env ID** | `textworld` | `fast_textworld` |
| **Speed** | ~200 steps/sec | ~8,000 steps/sec (30-50x faster) |
| **Dependencies** | `textworld` library + Inform7 | None (stdlib only) |
| **Game logic parity** | Reference | 100% (score, reward, done, won) |
| **Text parity** | Reference | 98% exact match (remaining 2% is Z-machine newline variance) |

## Quick Start

### Using FastTextWorldEnv in training

Register `fast_textworld` as the env ID in your config:

```yaml
env:
  id: fast_textworld
  max_turns: 50
  step_penalty: 0.0
  efficiency_bonus: 0.0
```

Pass `game_file` in extras:

```python
env = FastTextWorldEnv(env_config, extras={"game_file": "/path/to/game.json"})
```

The env accepts `.z8` paths too — it automatically resolves the corresponding `.json` file.

### Using FastTextWorldSimulator directly

```python
from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator

sim = FastTextWorldSimulator("path/to/game.json")
obs, info = sim.reset()
# info: {"score": 0, "max_score": 2, "won": False, "lost": False}

obs, reward, done, info = sim.step("go north")
obs, reward, done, info = sim.step("open chest")
obs, reward, done, info = sim.step("take key from chest")
```

### Interactive play / benchmarking

```bash
# Interactive play
python scripts/play_fast_sim.py /path/to/game.json

# Auto-play using the built-in walkthrough
python scripts/play_fast_sim.py /path/to/game.json --walkthrough

# Benchmark all games in a directory
python scripts/play_fast_sim.py /path/to/games_dir --benchmark
```

## Architecture

### FastTextWorldSimulator (`fast_sim.py`)

Pure-Python game engine (~1350 lines) that reads `.json` game specs directly. No subprocesses, no IPC, no global locks.

**Game state tracking:**
- Player location, inventory
- Container states (open/closed/locked)
- Door states (open/closed/locked)
- Object locations (rooms, containers, supporters, inventory)
- Key-lock matching
- Room connectivity and door links

**Supported commands:**
- Movement: `go north/south/east/west` (and bare direction words)
- Interaction: `open`, `close`, `lock`, `unlock`, `take`, `drop`, `put/insert`, `eat`, `examine/look at`
- Information: `look`, `inventory`, `examine`

**Text generation:**
- Inform7-compatible template resolution (`[if open]...[else if closed]...[end if]`)
- Context-aware conditionals (container/door descriptions resolve against the entity being described)
- "Locked implies closed" Inform7 semantics
- Single-quote to double-quote conversion (matching Z-machine output)
- Score announcements, end-game formatting

**Quest system:**
- Sequential incremental quests with precondition checking
- Actual reward values from quest spec (not hardcoded)
- Win event and fail event tracking
- Predicate verification: `at`, `in`, `on`, `open`, `closed`, `locked`, `eaten`, `link`, `match`

### FastTextWorldEnv (`fast_env.py`)

Drop-in replacement for `TextWorldEnv` (~340 lines). Same `BaseTextEnv` interface, same config knobs:

- `max_turns`, `step_penalty`, `efficiency_bonus`
- `memory_window`, `max_memory_docs`, `max_doc_tokens`
- Modality payload injection (placeholder tokens for memory documents)
- Action parsing: `[ACTION: command]` format with fallback heuristics

## Validation

Exhaustive comparison against the Inform7 reference engine across 75 games, 509 walkthrough steps:

| Metric | Match Rate |
|--------|-----------|
| Score | 509/509 (100%) |
| Reward | 509/509 (100%) |
| Done | 509/509 (100%) |
| Won | 509/509 (100%) |
| Text (exact) | 499/509 (98%) |

The 10 text mismatches are:
- 5 Inform7 Z-machine newline variance (non-deterministic paragraph spacing)
- 5 broken TextWorld-generated walkthroughs (commands reference entities in wrong rooms — both engines fail identically)

Tests: `tests/test_fast_sim_vs_inform7.py` (17 tests: 15 correctness + 2 speed benchmarks)

## When to use which

- **Training / data generation**: Use `fast_textworld`. The 30-50x speedup matters at scale.
- **Debugging / reference comparison**: Use `textworld` (original) when you need exact Inform7 output for debugging.
