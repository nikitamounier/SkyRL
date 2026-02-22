"""
Handlers for a "text modality" that re-embeds provided token ids with the base model's
embedding table and returns them unchanged. Useful for end-to-end multimodal tests without
introducing a new modality or extra trainable parameters.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch

from skyrl_train.modalities.checkpoint_utils import load_embedding_weight
from skyrl_train.modalities.handlers import ModalityEncoderProtocol, ModalityProjectorProtocol


class TokenIdListEncoder(ModalityEncoderProtocol):
    """
    Expects each payload to be a list of token ids. Returns a 2D LongTensor shaped
    (seq_len, 1) so the downstream projection can gather embeddings.
    """

    def __init__(self, modality_id: str, role: str, **_: Any):
        self.modality_id = modality_id
        self.role = role

    def encode(self, payloads: Sequence[Any]) -> Sequence[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for idx, payload in enumerate(payloads):
            if not isinstance(payload, (list, tuple)):
                raise TypeError(
                    f"Modality `{self.modality_id}` encoder expected list/tuple payload; "
                    f"got {type(payload).__name__} at index {idx}."
                )
            tensor = torch.tensor(payload, dtype=torch.long)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            outputs.append(tensor)
        return outputs


class EmbeddingLookupProjection(ModalityProjectorProtocol):
    """
    Looks up embeddings for token ids using the base model's embedding table. No trainable params.
    """

    DEFAULT_EMBEDDING_NAMES = (
        "model.embed_tokens.weight",
        "model.wte.weight",
    )

    def __init__(
        self,
        modality_id: str,
        role: str,
        model_path: str,
        embedding_weight_names: Optional[Sequence[str]] = None,
        **_: Any,
    ):
        self.modality_id = modality_id
        self.role = role
        self.model_path = model_path
        self.embedding_weight_names = list(embedding_weight_names or self.DEFAULT_EMBEDDING_NAMES)
        self.embedding_weight, discovered_name = load_embedding_weight(
            model_path=self.model_path,
            embedding_weight_names=self.embedding_weight_names,
            modality_id=self.modality_id,
        )
        self.embedding_weight_names = [discovered_name] + [
            existing for existing in self.embedding_weight_names if existing != discovered_name
        ]
        self.hidden_size = self.embedding_weight.shape[1]

    def project(self, features: torch.Tensor) -> torch.Tensor:
        # features is expected to be (seq_len, 1) token ids
        if features.dtype != torch.long:
            raise TypeError(
                f"Projection for `{self.modality_id}` expected token ids (LongTensor); got dtype {features.dtype}."
            )
        token_ids = features.squeeze(-1)
        embeddings = torch.embedding(self.embedding_weight, token_ids)
        return embeddings


__all__ = ["TokenIdListEncoder", "EmbeddingLookupProjection"]
