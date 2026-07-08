from __future__ import annotations

import glob
import os
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from loguru import logger
from safetensors import safe_open


def resolve_model_dir(model_path: str, *, modality_id: str | None = None) -> str:
    """Resolve a local checkpoint directory for a local path or HF repo id."""
    if os.path.isdir(model_path):
        return model_path

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_path,
            allow_patterns=("*.safetensors",),
        )
        if modality_id:
            logger.info(
                "Resolved repo id `{}` to local snapshot `{}` for modality `{}`.",
                model_path,
                local_dir,
                modality_id,
            )
        return local_dir
    except Exception:
        if modality_id:
            logger.debug(
                "Failed to resolve repo id `{}` for modality `{}`; using raw path.",
                model_path,
                modality_id,
            )
        return model_path


def find_safetensors_files(model_dir: str) -> list[str]:
    primary = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(primary):
        return [primary]

    shard_pattern = os.path.join(model_dir, "model-*.safetensors")
    return sorted(glob.glob(shard_pattern))


def load_embedding_weight(
    *,
    model_path: str,
    embedding_weight_names: Sequence[str],
    modality_id: str,
) -> Tuple[torch.Tensor, str]:
    """Load the first matching 2D embedding tensor from safetensors shards."""
    model_dir = resolve_model_dir(model_path, modality_id=modality_id)
    candidate_files = find_safetensors_files(model_dir)
    if not candidate_files:
        raise FileNotFoundError(
            f"No safetensors checkpoint found under `{model_dir}` for modality `{modality_id}`."
        )

    for file_path in candidate_files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for name in embedding_weight_names:
                if name not in handle.keys():
                    continue

                tensor = handle.get_tensor(name)
                if tensor.dim() != 2:
                    logger.warning(
                        "Ignoring embedding `{}` in `{}` due to unexpected shape {}.",
                        name,
                        file_path,
                        tuple(tensor.shape),
                    )
                    continue

                logger.info(
                    "Loaded embedding weight `{}` for modality `{}` from `{}` (shape={}).",
                    name,
                    modality_id,
                    file_path,
                    tuple(tensor.shape),
                )
                return tensor, name

    raise RuntimeError(
        f"Could not find embedding weights {list(embedding_weight_names)} in `{model_dir}` "
        f"for modality `{modality_id}`."
    )


def load_module_checkpoint(
    module: nn.Module,
    *,
    checkpoint_path: str,
    prefix: str,
    module_name: str,
) -> Tuple[list[str], list[str]]:
    """Load checkpoint weights into a module, optionally filtering by key prefix."""
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a state_dict.")

    filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not filtered:
        logger.warning(
            "No parameters matched prefix '{}' in checkpoint {}; loading full state_dict for {}.",
            prefix,
            checkpoint_path,
            module_name,
        )
        filtered = state_dict

    missing, unexpected = module.load_state_dict(filtered, strict=False)
    missing_keys = list(missing)
    unexpected_keys = list(unexpected)
    if missing_keys:
        logger.warning("Missing keys when loading {} checkpoint: {}", module_name, missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys when loading {} checkpoint: {}", module_name, unexpected_keys)
    return missing_keys, unexpected_keys


__all__ = [
    "find_safetensors_files",
    "load_embedding_weight",
    "load_module_checkpoint",
    "resolve_model_dir",
]
