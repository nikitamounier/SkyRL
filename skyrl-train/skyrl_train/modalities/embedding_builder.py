from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from loguru import logger

EmbeddingSpan = Tuple[int, int]  # (start, length)


class PromptEmbeddingBuilder:
    """Constructs prompt embeddings by combining base token embeddings and modality replacements."""

    DEFAULT_EMBEDDING_NAMES = (
        "model.embed_tokens.weight",
        "model.wte.weight",
    )

    def __init__(
        self,
        *,
        embedding_dim: int,
        target_device: torch.device,
        target_dtype: torch.dtype,
        embedding_weight_names: Sequence[str] | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.device = target_device
        self.dtype = target_dtype
        self.embedding_weight_names = list(embedding_weight_names or self.DEFAULT_EMBEDDING_NAMES)
        self._embedding_table: Optional[torch.Tensor] = None

    def has_base_embedding(self) -> bool:
        return self._embedding_table is not None

    def ensure_initialized(self, model_path: Optional[str]) -> None:
        if self._embedding_table is not None:
            return
        if model_path is None:
            logger.warning("PromptEmbeddingBuilder did not receive a model checkpoint path; waiting for weight sync.")
            return
        loaded = self._load_embedding_from_checkpoint(model_path)
        if loaded is not None:
            self._embedding_table = loaded.to(self.device, dtype=self.dtype)
            logger.info(
                "Loaded base embedding table from checkpoint `%s` with shape %s.",
                model_path,
                tuple(self._embedding_table.shape),
            )

    def _load_embedding_from_checkpoint(self, model_path: str) -> Optional[torch.Tensor]:
        def _find_safetensors(search_dir: str) -> List[str]:
            candidates: List[str] = []
            primary = os.path.join(search_dir, "model.safetensors")
            if os.path.isfile(primary):
                candidates.append(primary)
            else:
                shard_pattern = os.path.join(search_dir, "model-*.safetensors")
                candidates.extend(sorted(glob.glob(shard_pattern)))
            return candidates

        candidate_files: List[str] = []
        if os.path.isdir(model_path):
            candidate_files = _find_safetensors(model_path)

        resolved_dir: Optional[str] = None
        if not candidate_files:
            try:
                from huggingface_hub import snapshot_download

                resolved_dir = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=("*.safetensors",),
                )
                candidate_files = _find_safetensors(resolved_dir)
            except Exception:
                resolved_dir = None

        if not candidate_files:
            logger.warning(
                "PromptEmbeddingBuilder could not find safetensors checkpoint under `%s`. "
                "Expected `model.safetensors` or sharded files.",
                resolved_dir or model_path,
            )
            return None

        for file_path in candidate_files:
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for name in self.embedding_weight_names:
                    if name in handle.keys():
                        tensor = handle.get_tensor(name)
                        if tensor.dim() != 2:
                            logger.warning(
                                "Embedding tensor `%s` from `%s` has unexpected shape %s.",
                                name,
                                file_path,
                                tuple(tensor.shape),
                            )
                            continue
                        # prefer the discovered parameter name for future updates
                        self.embedding_weight_names = [name] + [
                            existing for existing in self.embedding_weight_names if existing != name
                        ]
                        return tensor
        logger.warning(
            "PromptEmbeddingBuilder did not find any of %s in checkpoint directory `%s`.",
            self.embedding_weight_names,
            model_path,
        )
        return None

    def update_from_named_weight(self, name: str, weight: torch.Tensor) -> None:
        if name not in self.embedding_weight_names:
            self.embedding_weight_names.append(name)
        self.set_base_embedding(weight)
        logger.info("Updated base embedding table from weight sync `%s`.", name)

    def set_base_embedding(self, weight: torch.Tensor) -> None:
        self._embedding_table = weight.detach().to(self.device, dtype=self.dtype)

    def gather_base_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding_table is None:
            raise RuntimeError(
                "PromptEmbeddingBuilder does not have a base embedding table. "
                "Ensure checkpoint loading or weight sync has occurred."
            )
        return torch.embedding(self._embedding_table, token_ids)

    def build_prompt_embeddings_for_batch(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        modality_replacements: Dict[int, List[Tuple[EmbeddingSpan, torch.Tensor]]],
    ) -> List[torch.Tensor]:
        # CRITICAL FIX: Wrap in torch.enable_grad() to ensure all embeddings get grad_fn
        # Even if they're built from detached base embeddings
        with torch.enable_grad():
            from loguru import logger
            outputs: List[torch.Tensor] = []
            for sample_idx, token_ids in enumerate(prompt_token_ids):
                token_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                # Mask out modality placeholder spans before embedding to avoid out-of-range ids
                for (start, length), _ in modality_replacements.get(sample_idx, []):
                    end = start + length
                    if start < 0 or end > token_tensor.shape[0]:
                        raise ValueError(
                            f"Replacement span ({start}, {length}) exceeds prompt length {token_tensor.shape[0]}."
                        )
                    token_tensor[start:end] = 0
                embeds = self.gather_base_embeddings(token_tensor)

                # If there are modality replacements, build the tensor by concatenating slices
                # to preserve gradient flow (in-place assignment breaks gradients)
                replacements = modality_replacements.get(sample_idx, [])

                logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: {len(replacements)} modality replacements")
                logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: Base embeds grad_fn: {type(embeds.grad_fn).__name__ if embeds.grad_fn else 'NONE'}")

                if replacements:
                    # Validate all replacements first
                    for (start, length), modality_tensor in replacements:
                        end = start + length
                        if end > embeds.shape[0]:
                            raise ValueError(
                                f"Replacement span ({start}, {length}) exceeds prompt length {embeds.shape[0]}."
                            )
                        if modality_tensor.shape[1] != embeds.shape[1]:
                            raise ValueError(
                                "Projection hidden size mismatch. Expected %d, got %d."
                                % (embeds.shape[1], modality_tensor.shape[1])
                            )

                    # Build the final tensor by concatenating base and modality embeddings
                    # Sort replacements by start position to process them in order
                    sorted_replacements = sorted(replacements, key=lambda x: x[0][0])
                    parts = []
                    last_end = 0

                    for idx, ((start, length), modality_tensor) in enumerate(sorted_replacements):
                        end = start + length
                        # Add base embeddings before this modality span
                        if start > last_end:
                            parts.append(embeds[last_end:start])
                        # Add modality embeddings
                        parts.append(modality_tensor[:length])
                        last_end = end

                    # Add remaining base embeddings after the last modality span
                    if last_end < embeds.shape[0]:
                        parts.append(embeds[last_end:])

                    embeds = torch.cat(parts, dim=0)
                    logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: After concat, grad_fn: {type(embeds.grad_fn).__name__ if embeds.grad_fn else 'NONE'}")
                else:
                    # CRITICAL FIX: If no modality embeddings, add dummy gradient
                    # Base embeddings are detached, so multiply by trainable scalar to give grad_fn
                    logger.warning(f"[EMBEDDING BUILDER] Sample {sample_idx}: NO modality replacements - adding dummy gradient")
                    if embeds.grad_fn is None:
                        logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: Base embeds have NO grad_fn, multiplying by dummy_scale")
                        dummy_scale = torch.ones(1, requires_grad=True, device=embeds.device, dtype=embeds.dtype)
                        embeds = embeds * dummy_scale
                        logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: After dummy_scale, grad_fn: {type(embeds.grad_fn).__name__ if embeds.grad_fn else 'STILL NONE!'}")
                    else:
                        logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: Base embeds already have grad_fn: {type(embeds.grad_fn).__name__}")

                # Final check before appending
                if embeds.grad_fn is None:
                    logger.error(f"[EMBEDDING BUILDER] Sample {sample_idx}: FINAL embeds have NO grad_fn - backward will fail!")
                else:
                    logger.info(f"[EMBEDDING BUILDER] Sample {sample_idx}: FINAL embeds have grad_fn: {type(embeds.grad_fn).__name__}")

                outputs.append(embeds)
        return outputs


__all__ = ["PromptEmbeddingBuilder", "EmbeddingSpan"]
