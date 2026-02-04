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
            for (start, length), modality_tensor in modality_replacements.get(sample_idx, []):
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
                embeds[start:end, :] = modality_tensor[:length]
            outputs.append(embeds)
        return outputs


__all__ = ["PromptEmbeddingBuilder", "EmbeddingSpan"]
