from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from loguru import logger

from skyrl_train.modalities.checkpoint_utils import load_embedding_weight

EmbeddingSpan = Tuple[int, int]  # (start, length)


class PromptEmbeddingBuilder:
    """Constructs prompt embeddings by combining base token embeddings and modality replacements."""

    DEFAULT_EMBEDDING_NAMES = (
        "model.embed_tokens.weight",
        "model.wte.weight",
        # qwen3.5 (Qwen3_5ForConditionalGeneration) nests the text embedding here
        "language_model.model.embed_tokens.weight",
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

        try:
            loaded, name = load_embedding_weight(
                model_path=model_path,
                embedding_weight_names=self.embedding_weight_names,
                modality_id="base_prompt",
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("PromptEmbeddingBuilder failed to load embedding checkpoint from `{}`: {}", model_path, exc)
            return

        # Prefer the discovered parameter name for future updates.
        self.embedding_weight_names = [name] + [existing for existing in self.embedding_weight_names if existing != name]
        if loaded.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Loaded embedding dimension mismatch for `{model_path}`: "
                f"expected {self.embedding_dim}, got {loaded.shape[1]}."
            )
        self._embedding_table = loaded.to(self.device, dtype=self.dtype)
        logger.info(
            "Loaded base embedding table from checkpoint `{}` with shape {}.",
            model_path,
            tuple(self._embedding_table.shape),
        )

    def update_from_named_weight(self, name: str, weight: torch.Tensor) -> None:
        if name not in self.embedding_weight_names:
            self.embedding_weight_names.append(name)
        self.set_base_embedding(weight)
        logger.info("Updated base embedding table from weight sync `{}`.", name)

    def set_base_embedding(self, weight: torch.Tensor) -> None:
        if weight.dim() != 2:
            raise ValueError(f"Base embedding weight must be 2D, got shape {tuple(weight.shape)}.")
        if weight.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Base embedding hidden size mismatch: expected {self.embedding_dim}, got {weight.shape[1]}."
            )
        self._embedding_table = weight.detach().to(self.device, dtype=self.dtype)

    def gather_base_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding_table is None:
            raise RuntimeError(
                "PromptEmbeddingBuilder does not have a base embedding table. "
                "Ensure checkpoint loading or weight sync has occurred."
            )
        return torch.embedding(self._embedding_table, token_ids)

    def _normalize_replacements(
        self,
        *,
        sample_idx: int,
        replacements: Sequence[Tuple[EmbeddingSpan, torch.Tensor]],
        prompt_length: int,
    ) -> List[Tuple[int, int, torch.Tensor]]:
        normalized: List[Tuple[int, int, torch.Tensor]] = []
        for (start, length), modality_tensor in replacements:
            end = start + length
            if start < 0 or length <= 0 or end > prompt_length:
                raise ValueError(
                    f"Replacement span ({start}, {length}) exceeds prompt length {prompt_length} "
                    f"for sample {sample_idx}."
                )
            if modality_tensor.dim() != 2:
                raise ValueError(
                    f"Modality replacement tensor for sample {sample_idx} must be 2D; "
                    f"got shape {tuple(modality_tensor.shape)}."
                )
            if modality_tensor.shape[0] != length:
                raise ValueError(
                    f"Replacement length mismatch for sample {sample_idx}: expected {length}, "
                    f"got {modality_tensor.shape[0]}."
                )
            if modality_tensor.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Projection hidden size mismatch for sample {sample_idx}: expected {self.embedding_dim}, "
                    f"got {modality_tensor.shape[1]}."
                )
            normalized.append(
                (
                    start,
                    end,
                    modality_tensor.to(device=self.device, dtype=self.dtype),
                )
            )

        normalized.sort(key=lambda item: item[0])
        prev_end = 0
        for start, end, _ in normalized:
            if start < prev_end:
                raise ValueError(
                    f"Overlapping modality replacement spans detected for sample {sample_idx}: "
                    f"start={start} overlaps previous end={prev_end}."
                )
            prev_end = end
        return normalized

    def _apply_replacements(
        self,
        *,
        embeds: torch.Tensor,
        replacements: Sequence[Tuple[int, int, torch.Tensor]],
    ) -> torch.Tensor:
        if not replacements:
            return embeds

        parts: List[torch.Tensor] = []
        cursor = 0
        for start, end, modality_tensor in replacements:
            if start > cursor:
                parts.append(embeds[cursor:start])
            parts.append(modality_tensor)
            cursor = end

        if cursor < embeds.shape[0]:
            parts.append(embeds[cursor:])
        return torch.cat(parts, dim=0)

    def build_prompt_embeddings_for_batch(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        modality_replacements: Dict[int, List[Tuple[EmbeddingSpan, torch.Tensor]]],
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for sample_idx, token_ids in enumerate(prompt_token_ids):
            token_tensor = torch.as_tensor(token_ids, device=self.device, dtype=torch.long)
            normalized_replacements = self._normalize_replacements(
                sample_idx=sample_idx,
                replacements=modality_replacements.get(sample_idx, []),
                prompt_length=token_tensor.shape[0],
            )

            # Mask placeholder spans to avoid invalid IDs before lookup.
            if normalized_replacements:
                token_tensor = token_tensor.clone()
                for start, end, _ in normalized_replacements:
                    token_tensor[start:end] = 0

            embeds = self.gather_base_embeddings(token_tensor)
            embeds = self._apply_replacements(embeds=embeds, replacements=normalized_replacements)
            if embeds.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding hidden size mismatch for sample {sample_idx}: expected {self.embedding_dim}, "
                    f"got {embeds.shape[1]}."
                )
            outputs.append(embeds)
        return outputs


__all__ = ["PromptEmbeddingBuilder", "EmbeddingSpan"]
