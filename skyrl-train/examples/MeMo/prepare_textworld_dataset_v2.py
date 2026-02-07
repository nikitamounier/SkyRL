#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import datasets

_LOCAL_BIN = str(Path.home() / ".local" / "bin")
os.environ["PATH"] = f"{_LOCAL_BIN}:{os.environ.get('PATH', '')}"


@dataclass(frozen=True)
class DifficultyPreset:
    world_size: Tuple[int, int]
    nb_objects: Tuple[int, int]
    quest_length: Tuple[int, int]


@dataclass(frozen=True)
class GameSpec:
    game_index: int
    game_seed: int
    difficulty: str
    world_size: int
    nb_objects: int
    quest_length: int
    game_file: Path


DIFFICULTY_PRESETS: Dict[str, DifficultyPreset] = {
    "easy": DifficultyPreset(world_size=(10, 16), nb_objects=(8, 14), quest_length=(3, 6)),
    "medium": DifficultyPreset(world_size=(15, 25), nb_objects=(12, 22), quest_length=(6, 10)),
    "hard": DifficultyPreset(world_size=(24, 40), nb_objects=(20, 35), quest_length=(10, 16)),
}


PROMPT_VARIANTS: Dict[str, Dict[str, str]] = {
    "v1_strict_action": {
        "system": (
            "You are an agent in a text adventure game.\n"
            "Think briefly, then ALWAYS end with exactly one action line in this format:\n"
            "[ACTION: <command>]\n"
            "Use short game commands (look, inventory, examine, take, open, close, go north, etc.).\n"
            "Memory slots from prior turns:\n{memory_slots}"
        ),
        "user": "Start the game and choose the next best action.",
    },
    "v2_memory_first": {
        "system": (
            "You play TextWorld efficiently.\n"
            "Use memory summaries when useful and avoid repeating failed actions.\n"
            "Return one final action command in the format [ACTION: <command>].\n"
            "Memory slots:\n{memory_slots}"
        ),
        "user": "What is your next action?",
    },
    "v3_compact": {
        "system": (
            "Solve the game step by step.\n"
            "Output must end with [ACTION: <command>].\n"
            "Memory:\n{memory_slots}"
        ),
        "user": "Play optimally.",
    },
    "v4_safety_parseable": {
        "system": (
            "You are playing a text-based game.\n"
            "Final line must be parseable as [ACTION: ...] with only a single command.\n"
            "Do not output multiple actions.\n"
            "Memory context:\n{memory_slots}"
        ),
        "user": "Continue the game.",
    },
}


def _parse_weighted_mix(raw: str, valid_labels: Iterable[str]) -> Dict[str, float]:
    valid = set(valid_labels)
    result: Dict[str, float] = {}
    for part in raw.split(","):
        entry = part.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid difficulty mix entry `{entry}`. Expected format `label:weight`."
            )
        label, weight_raw = entry.split(":", 1)
        label = label.strip().lower()
        if label not in valid:
            raise ValueError(
                f"Unknown difficulty `{label}` in mix. Valid choices: {sorted(valid)}."
            )
        try:
            weight = float(weight_raw.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid weight `{weight_raw}` for difficulty `{label}`.") from exc
        if weight <= 0:
            raise ValueError(f"Weight for `{label}` must be > 0. Got {weight}.")
        result[label] = weight

    if not result:
        raise ValueError("Difficulty mix must include at least one non-zero entry.")

    total = sum(result.values())
    for key in list(result.keys()):
        result[key] = result[key] / total
    return result


def _choose_weighted_label(
    rng: random.Random, weighted: Mapping[str, float], labels_in_order: Sequence[str]
) -> str:
    # Keep this deterministic and Python-version-stable.
    threshold = rng.random()
    cumulative = 0.0
    for label in labels_in_order:
        if label not in weighted:
            continue
        cumulative += weighted[label]
        if threshold <= cumulative:
            return label
    return labels_in_order[-1]


def _build_memory_slots(placeholder_token: str, max_memory_docs: int) -> str:
    return " ".join([placeholder_token] * max_memory_docs)


def _build_prompt(
    *,
    prompt_variant_id: str,
    placeholder_token: str,
    max_memory_docs: int,
) -> List[Dict[str, str]]:
    if prompt_variant_id not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant_id `{prompt_variant_id}`. "
            f"Choices: {sorted(PROMPT_VARIANTS.keys())}"
        )
    template = PROMPT_VARIANTS[prompt_variant_id]
    memory_slots = _build_memory_slots(placeholder_token, max_memory_docs)
    return [
        {
            "role": "system",
            "content": template["system"].format(memory_slots=memory_slots),
        },
        {"role": "user", "content": template["user"]},
    ]


def _sample_param(rng: random.Random, bounds: Tuple[int, int]) -> int:
    lo, hi = bounds
    if lo > hi:
        raise ValueError(f"Invalid bounds: ({lo}, {hi})")
    return rng.randint(lo, hi)


def _sample_game_specs(
    *,
    num_games: int,
    seed_start: int,
    rng: random.Random,
    weighted_mix: Mapping[str, float],
    games_dir: Path,
) -> List[GameSpec]:
    labels = [label for label in DIFFICULTY_PRESETS.keys() if label in weighted_mix]
    specs: List[GameSpec] = []
    for game_idx in range(num_games):
        difficulty = _choose_weighted_label(rng, weighted_mix, labels)
        preset = DIFFICULTY_PRESETS[difficulty]
        world_size = _sample_param(rng, preset.world_size)
        nb_objects = _sample_param(rng, preset.nb_objects)
        quest_length = _sample_param(rng, preset.quest_length)
        game_seed = seed_start + 9973 * (game_idx + 1)
        game_file = games_dir / (
            f"game_{game_idx:05d}_{difficulty}_w{world_size}_o{nb_objects}_q{quest_length}_s{game_seed}.z8"
        )
        specs.append(
            GameSpec(
                game_index=game_idx,
                game_seed=game_seed,
                difficulty=difficulty,
                world_size=world_size,
                nb_objects=nb_objects,
                quest_length=quest_length,
                game_file=game_file,
            )
        )
    return specs


def _generate_games(
    *,
    specs: Sequence[GameSpec],
    overwrite: bool,
) -> List[GameSpec]:
    generated: List[GameSpec] = []
    for spec in specs:
        spec.game_file.parent.mkdir(parents=True, exist_ok=True)
        if spec.game_file.exists() and not overwrite:
            generated.append(spec)
            continue
        cmd = [
            "tw-make",
            "custom",
            "--world-size",
            str(spec.world_size),
            "--nb-objects",
            str(spec.nb_objects),
            "--quest-length",
            str(spec.quest_length),
            "--seed",
            str(spec.game_seed),
            "--output",
            str(spec.game_file),
        ]
        try:
            subprocess.run(cmd, check=True)
            generated.append(spec)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] Failed to generate game {spec.game_file.name}: {exc}")
    return generated


def _split_games(
    specs: Sequence[GameSpec],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[GameSpec]]:
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio must be > 0 and val_ratio must be >= 0.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    specs_list = list(specs)
    rng = random.Random(seed)
    rng.shuffle(specs_list)

    total = len(specs_list)
    if total == 0:
        return {"train": [], "validation": [], "test": []}

    train_end = max(1, int(total * train_ratio))
    val_end = train_end + int(total * val_ratio)

    train_specs = specs_list[:train_end]
    val_specs = specs_list[train_end:val_end]
    test_specs = specs_list[val_end:]

    # Ensure validation has at least one game when possible.
    if not val_specs and len(train_specs) > 1:
        val_specs = [train_specs.pop()]
    # Keep test optional; if empty, that's acceptable.

    return {"train": train_specs, "validation": val_specs, "test": test_specs}


def _choose_prompt_variant_for_episode(
    *,
    rng: random.Random,
    prompt_variant_ids: Sequence[str],
    game_seed: int,
    episode_index: int,
) -> str:
    # Deterministic variant assignment per (game, episode), independent of row order.
    local_rng = random.Random((game_seed << 8) + episode_index + 17)
    _ = rng  # Keep signature symmetric with other helper calls.
    return prompt_variant_ids[local_rng.randrange(len(prompt_variant_ids))]


def _make_rows(
    *,
    split_name: str,
    specs: Sequence[GameSpec],
    episodes_per_game: int,
    placeholder_token: str,
    max_memory_docs: int,
    prompt_variant_ids: Sequence[str],
    seed: int,
) -> List[Dict[str, object]]:
    if episodes_per_game <= 0:
        raise ValueError("episodes_per_game must be > 0.")
    if not prompt_variant_ids:
        raise ValueError("At least one prompt variant must be selected.")

    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    row_index = 0
    for spec in specs:
        game_id = spec.game_file.stem
        for episode_index in range(episodes_per_game):
            prompt_variant_id = _choose_prompt_variant_for_episode(
                rng=rng,
                prompt_variant_ids=prompt_variant_ids,
                game_seed=spec.game_seed,
                episode_index=episode_index,
            )
            row = {
                "prompt": _build_prompt(
                    prompt_variant_id=prompt_variant_id,
                    placeholder_token=placeholder_token,
                    max_memory_docs=max_memory_docs,
                ),
                "env_class": "textworld",
                "game_file": str(spec.game_file),
                "extra_info": {
                    "split": split_name,
                    "row_index": row_index,
                    "episode_index": episode_index,
                    "game_index": spec.game_index,
                    "game_id": game_id,
                    "game_seed": spec.game_seed,
                    "difficulty": spec.difficulty,
                    "world_size": spec.world_size,
                    "nb_objects": spec.nb_objects,
                    "quest_length": spec.quest_length,
                    "prompt_variant_id": prompt_variant_id,
                    "dataset_version": "textworld_memo_v2",
                },
                "modalities": {
                    "memo_memory": [[] for _ in range(max_memory_docs)],
                },
            }
            rows.append(row)
            row_index += 1
    return rows


def _write_parquet(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = datasets.Dataset.from_list(list(rows))
    ds.to_parquet(str(output_path))


def _build_stats(
    *,
    split_to_specs: Mapping[str, Sequence[GameSpec]],
    split_to_rows: Mapping[str, Sequence[Mapping[str, object]]],
    config: Mapping[str, object],
) -> Dict[str, object]:
    stats: Dict[str, object] = {"config": dict(config), "splits": {}}
    for split_name in ("train", "validation", "test"):
        specs = list(split_to_specs.get(split_name, []))
        rows = list(split_to_rows.get(split_name, []))
        by_difficulty: Dict[str, int] = {}
        for spec in specs:
            by_difficulty[spec.difficulty] = by_difficulty.get(spec.difficulty, 0) + 1

        prompt_counts: Dict[str, int] = {}
        for row in rows:
            extra = row.get("extra_info", {})
            if isinstance(extra, dict):
                variant = str(extra.get("prompt_variant_id", "unknown"))
            else:
                variant = "unknown"
            prompt_counts[variant] = prompt_counts.get(variant, 0) + 1

        avg_quest_length = 0.0
        if specs:
            avg_quest_length = sum(spec.quest_length for spec in specs) / len(specs)

        stats["splits"][split_name] = {
            "num_games": len(specs),
            "num_rows": len(rows),
            "difficulty_counts": by_difficulty,
            "prompt_variant_counts": prompt_counts,
            "avg_quest_length": avg_quest_length,
        }
    return stats


def _write_stats(stats: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_prompt_variants(raw: str) -> List[str]:
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    if not selected:
        raise ValueError("At least one prompt variant must be selected.")
    invalid = [name for name in selected if name not in PROMPT_VARIANTS]
    if invalid:
        raise ValueError(
            f"Unknown prompt variants {invalid}. Available: {sorted(PROMPT_VARIANTS.keys())}"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a robust TextWorld dataset for MeMo memory-module training."
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/textworld_memo_v2",
        help="Output directory for parquet files and stats.",
    )
    parser.add_argument(
        "--games_dir",
        default=None,
        help="Directory for generated .z8 games. Defaults to <output_dir>/games.",
    )
    parser.add_argument("--num_games", type=int, default=200, help="Number of unique games to generate.")
    parser.add_argument(
        "--episodes_per_game",
        type=int,
        default=3,
        help="How many dataset rows to emit per unique game.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Base RNG seed.")
    parser.add_argument(
        "--difficulty_mix",
        default="easy:0.35,medium:0.45,hard:0.20",
        help="Weighted mix like 'easy:0.4,medium:0.4,hard:0.2'.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train split ratio by games.")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation split ratio by games.")
    parser.add_argument(
        "--prompt_variants",
        default="v1_strict_action,v2_memory_first,v3_compact,v4_safety_parseable",
        help="Comma-separated prompt variant ids.",
    )
    parser.add_argument("--placeholder_token", default="<|image_pad|>", help="Modality placeholder token.")
    parser.add_argument(
        "--max_memory_docs",
        type=int,
        default=20,
        help="Number of memory doc placeholders pre-allocated in prompt + payload scaffold.",
    )
    parser.add_argument("--overwrite_games", action="store_true", help="Regenerate games even if files exist.")
    parser.add_argument(
        "--skip_test_split_file",
        action="store_true",
        help="Do not write test.parquet even if test split exists.",
    )
    args = parser.parse_args()

    if args.num_games <= 0:
        raise ValueError("--num_games must be > 0.")
    if args.max_memory_docs <= 0:
        raise ValueError("--max_memory_docs must be > 0.")

    output_dir = Path(os.path.expanduser(args.output_dir))
    games_dir = (
        Path(os.path.expanduser(args.games_dir))
        if args.games_dir
        else output_dir / "games"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir.mkdir(parents=True, exist_ok=True)

    prompt_variant_ids = _parse_prompt_variants(args.prompt_variants)
    weighted_mix = _parse_weighted_mix(args.difficulty_mix, DIFFICULTY_PRESETS.keys())
    rng = random.Random(args.seed)

    sampled_specs = _sample_game_specs(
        num_games=args.num_games,
        seed_start=args.seed,
        rng=rng,
        weighted_mix=weighted_mix,
        games_dir=games_dir,
    )
    generated_specs = _generate_games(specs=sampled_specs, overwrite=args.overwrite_games)
    if not generated_specs:
        raise RuntimeError(
            "No games were generated successfully. Verify `tw-make` is installed and on PATH."
        )

    split_to_specs = _split_games(
        generated_specs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_to_rows: Dict[str, List[Dict[str, object]]] = {}
    for split_name in ("train", "validation", "test"):
        split_specs = split_to_specs.get(split_name, [])
        split_to_rows[split_name] = _make_rows(
            split_name=split_name,
            specs=split_specs,
            episodes_per_game=args.episodes_per_game,
            placeholder_token=args.placeholder_token,
            max_memory_docs=args.max_memory_docs,
            prompt_variant_ids=prompt_variant_ids,
            seed=args.seed + 37,
        )

    _write_parquet(split_to_rows["train"], output_dir / "train.parquet")
    _write_parquet(split_to_rows["validation"], output_dir / "validation.parquet")
    if split_to_rows["test"] and not args.skip_test_split_file:
        _write_parquet(split_to_rows["test"], output_dir / "test.parquet")

    config_for_stats = {
        "seed": args.seed,
        "num_games_requested": args.num_games,
        "num_games_generated": len(generated_specs),
        "episodes_per_game": args.episodes_per_game,
        "difficulty_mix": weighted_mix,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "prompt_variants": prompt_variant_ids,
        "placeholder_token": args.placeholder_token,
        "max_memory_docs": args.max_memory_docs,
        "output_dir": str(output_dir),
        "games_dir": str(games_dir),
    }
    stats = _build_stats(
        split_to_specs=split_to_specs,
        split_to_rows=split_to_rows,
        config=config_for_stats,
    )
    _write_stats(stats, output_dir / "dataset_stats.json")

    print(
        f"Wrote dataset to {output_dir}: "
        f"{len(split_to_rows['train'])} train rows, "
        f"{len(split_to_rows['validation'])} validation rows, "
        f"{len(split_to_rows['test'])} test rows."
    )
    print(f"Generated unique games: {len(generated_specs)}")
    print(f"Prompt variants: {', '.join(prompt_variant_ids)}")
    print(f"Stats file: {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
