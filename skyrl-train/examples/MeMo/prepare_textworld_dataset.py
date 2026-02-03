#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

_LOCAL_BIN = str(Path.home() / ".local" / "bin")
os.environ["PATH"] = f"{_LOCAL_BIN}:{os.environ.get('PATH', '')}"


SIZE_PRESETS: Dict[str, Tuple[int, int, int]] = {
    "TINY": (10, 15, 5),
    "SMALL": (15, 20, 10),
    "MEDIUM": (20, 30, 20),
    "LARGE": (30, 50, 30),
}


def _generate_games(
    games_dir: Path,
    size: str,
    num_games: int,
    seed_start: int,
    overwrite: bool,
) -> List[Path]:
    games_dir.mkdir(parents=True, exist_ok=True)
    if size not in SIZE_PRESETS:
        raise ValueError(f"Unknown size preset {size}. Choose from {list(SIZE_PRESETS)}.")
    world_size, nb_objects, quest_length = SIZE_PRESETS[size]

    generated: List[Path] = []
    for idx in range(1, num_games + 1):
        seed = seed_start + idx
        out_file = games_dir / f"game_{size.lower()}_{idx}.z8"
        if out_file.exists() and not overwrite:
            generated.append(out_file)
            continue
        cmd = [
            "tw-make",
            "custom",
            "--world-size",
            str(world_size),
            "--nb-objects",
            str(nb_objects),
            "--quest-length",
            str(quest_length),
            "--seed",
            str(seed),
            "--output",
            str(out_file),
        ]
        subprocess.run(cmd, check=True)
        generated.append(out_file)
    return generated


def _build_prompt(placeholder_token: str, max_memory_docs: int) -> List[Dict[str, str]]:
    placeholder_block = " ".join([placeholder_token] * max_memory_docs)
    system_text = (
        "You are playing a text-based adventure game. "
        "Respond with a single action command (e.g., 'look', 'go north', 'take key').\n"
        f"Memory slots: {placeholder_block}"
    )
    return [{"role": "system", "content": system_text}]


def _make_row(
    game_file: Path,
    placeholder_token: str,
    max_memory_docs: int,
    split: str,
    index: int,
) -> Dict[str, object]:
    return {
        "prompt": _build_prompt(placeholder_token, max_memory_docs),
        "env_class": "textworld",
        "game_file": str(game_file),
        "extra_info": {
            "split": split,
            "index": index,
            "game_file": str(game_file),
        },
        "modalities": {
            "memo_memory": [[] for _ in range(max_memory_docs)],
        },
    }


def _write_dataset(rows: List[Dict[str, object]], output_path: Path) -> None:
    dataset = datasets.Dataset.from_list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TextWorld dataset for MeMo training in SkyRL.")
    parser.add_argument("--output_dir", default="~/data/textworld_memo", help="Output directory.")
    parser.add_argument("--games_dir", default=None, help="Directory containing .z8 games.")
    parser.add_argument("--size", default="SMALL", choices=list(SIZE_PRESETS))
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--placeholder_token", default="<|image_pad|>")
    parser.add_argument("--max_memory_docs", type=int, default=4)
    parser.add_argument("--overwrite_games", action="store_true")
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))
    games_dir = Path(os.path.expanduser(args.games_dir)) if args.games_dir else output_dir / "games"

    game_files = _generate_games(
        games_dir=games_dir,
        size=args.size.upper(),
        num_games=args.num_games,
        seed_start=args.seed,
        overwrite=args.overwrite_games,
    )

    random.seed(args.seed)
    random.shuffle(game_files)

    split_idx = max(1, int(len(game_files) * args.train_ratio))
    train_files = game_files[:split_idx]
    val_files = game_files[split_idx:] or game_files[:1]

    train_rows = [
        _make_row(game_file, args.placeholder_token, args.max_memory_docs, "train", idx)
        for idx, game_file in enumerate(train_files)
    ]
    val_rows = [
        _make_row(game_file, args.placeholder_token, args.max_memory_docs, "validation", idx)
        for idx, game_file in enumerate(val_files)
    ]

    _write_dataset(train_rows, output_dir / "train.parquet")
    _write_dataset(val_rows, output_dir / "validation.parquet")

    print(f"Wrote {len(train_rows)} train and {len(val_rows)} validation samples to {output_dir}.")
    print(f"Games directory: {games_dir}")


if __name__ == "__main__":
    main()
