"""
Build a SkyRL cell-pathway dataset from the BioReasonCell v7.5 50k eval data.

Each sample carries the SFT ``question`` prompt with a ``<|cell_pad|>`` placeholder
prepended inside a ``Cell A: <|CELL_START|><|cell_pad|><|CELL_END|>`` block (byte-identical
to the BioReasonCell collator), plus the per-sample 2058-d STATE mean-embedding as the
``state_mod`` modality payload and the ground-truth ``pathway_change`` as the reward target.

The STATE embedding is looked up per ``cell_file`` stem from the per-file ``.pt`` store; rows
whose cell embedding is missing are dropped. Reads the cached HF Arrow files directly
(the shared HF cache is read-only), so no hub download / lockfile is needed.

Example::

    python examples/modalities/prepare_cell_pathway_data.py \
        --arrow_dir /large_storage/goodarzilab/bioreason_cell/.hf_home/datasets/wanglab___bio_reason_cell-experiment_data/sft_gene_pathway_v7_5_50k_full/0.0.0/53f2244edf02ff392de6e1445199a27b51fa5fc2 \
        --cell_embed_dir /large_storage/goodarzilab/bioreason_cell/embeddings/genetic_v7_5/cells \
        --output_dir ~/data/cell_pathway_v7_5 \
        --n_train 512 --n_val 256
"""

import argparse
import glob
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset

PLACEHOLDER_TOKEN = "<|cell_pad|>"
CELL_BLOCK = f"Cell A: <|CELL_START|>{PLACEHOLDER_TOKEN}<|CELL_END|>"
CLASSES = ("upregulated", "downregulated", "unchanged")


def load_split(arrow_dir: str, pattern: str) -> Dataset:
    files = sorted(glob.glob(os.path.join(arrow_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No arrow files matching {pattern!r} under {arrow_dir}")
    from datasets import concatenate_datasets

    return concatenate_datasets([Dataset.from_file(f) for f in files])


def build_sample(row: Dict[str, Any], idx: int, split: str, embed_vec: List[float]) -> Dict[str, Any]:
    gt = str(row["pathway_change"]).strip().lower()
    content = f"{CELL_BLOCK}\n\n{row['question']}"
    return {
        "data_source": "wanglab/BioReasonCell-ExperimentData",
        "prompt": [{"role": "user", "content": content}],
        "env_class": "cell_pathway",
        "reward_spec": {"method": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": idx,
            "sample_id": row.get("sample_id"),
            "dataset": row.get("dataset"),
            "gene_target": row.get("gene_target"),
            "cell_type": row.get("cell_type"),
            "pathway_name": row.get("pathway_name"),
            "cell_file": row.get("cell_file"),
        },
        # One <|cell_pad|> occurrence -> a single-element list holding the 2058-d STATE vector.
        "modalities": {"state_mod": [embed_vec]},
    }


def process_split(
    ds: Dataset,
    split_name: str,
    cell_embed_dir: str,
    output_path: str,
    limit: Optional[int],
) -> int:
    cache: Dict[str, Optional[List[float]]] = {}

    def get_vec(cell_file: str) -> Optional[List[float]]:
        if cell_file not in cache:
            path = os.path.join(cell_embed_dir, f"{cell_file}.pt")
            if os.path.isfile(path):
                emb = torch.load(path, map_location="cpu", weights_only=False)
                if emb.ndim > 1:
                    emb = emb.reshape(-1)
                cache[cell_file] = emb.to(dtype=torch.float32).tolist()
            else:
                cache[cell_file] = None
        return cache[cell_file]

    samples: List[Dict[str, Any]] = []
    dropped_missing = 0
    dropped_label = 0
    for idx, row in enumerate(ds):
        gt = str(row["pathway_change"]).strip().lower()
        if gt not in CLASSES:
            dropped_label += 1
            continue
        vec = get_vec(row["cell_file"])
        if vec is None:
            dropped_missing += 1
            continue
        samples.append(build_sample(row, idx, split_name, vec))
        if limit and len(samples) >= limit:
            break

    out = Dataset.from_list(samples)
    out.to_parquet(output_path)
    print(
        f"[{split_name}] wrote {len(samples)} rows -> {output_path} "
        f"(dropped {dropped_missing} missing-embedding, {dropped_label} bad-label)"
    )
    return len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow_dir", required=True, help="Dir with cached BioReasonCell-ExperimentData *.arrow files")
    parser.add_argument("--cell_embed_dir", required=True, help="Dir of per-cell STATE .pt embeddings")
    parser.add_argument("--output_dir", default="~/data/cell_pathway_v7_5")
    parser.add_argument("--n_train", type=int, default=512, help="0 = all")
    parser.add_argument("--n_val", type=int, default=256, help="0 = all")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # v7.5 test == val (copy); train is the disjoint 47.7k split.
    val_ds = load_split(args.arrow_dir, "*validation*.arrow")
    train_ds = load_split(args.arrow_dir, "*train*.arrow")

    process_split(
        train_ds, "train", args.cell_embed_dir,
        os.path.join(output_dir, "train.parquet"), args.n_train or None,
    )
    process_split(
        val_ds, "validation", args.cell_embed_dir,
        os.path.join(output_dir, "validation.parquet"), args.n_val or None,
    )
    print(f"Done. Output in {output_dir}")


if __name__ == "__main__":
    main()
