"""
Handlers for a "text modality" that re-embeds provided token ids with the base model's
embedding table and returns them unchanged. Useful for end-to-end multimodal tests without
introducing a new modality or extra trainable parameters.
"""

from __future__ import annotations

import glob
import os
from typing import Any, List, Optional, Sequence

import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.embedding_weight = self._load_embedding_weight()
        if self.embedding_weight is None:
            raise RuntimeError(
                f"Failed to load embedding weight for modality `{modality_id}` from `{model_path}`."
            )
        self.hidden_size = self.embedding_weight.shape[1]

    def _resolve_model_dir(self) -> str:
        if os.path.isdir(self.model_path):
            return self.model_path
        try:
            local_dir = snapshot_download(
                repo_id=self.model_path,
                allow_patterns=("*.safetensors",),  # only what we need
            )
            logger.info(
                "Resolved repo id `%s` to local snapshot `%s` for modality `%s`.",
                self.model_path,
                local_dir,
                self.modality_id,
            )
            return local_dir
        except Exception:
            logger.exception(
                "Failed to resolve repo id `%s`; falling back to raw path.",
                self.model_path,
            )
            return self.model_path

    def _load_embedding_weight(self) -> Optional[torch.Tensor]:
        model_dir = self._resolve_model_dir()

        candidate_files: List[str] = []
        primary = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(primary):
            candidate_files.append(primary)
        else:
            shard_pattern = os.path.join(model_dir, "model-*.safetensors")
            candidate_files.extend(sorted(glob.glob(shard_pattern)))

        if not candidate_files:
            logger.error(
                "No safetensors checkpoint found under `%s` while initializing projection for modality `%s`.",
                model_dir,
                self.modality_id,
            )
            return None

        for path in candidate_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in self.embedding_weight_names:
                    if name in handle.keys():
                        tensor = handle.get_tensor(name)
                        if tensor.dim() != 2:
                            logger.warning(
                                "Ignoring embedding `%s` in `%s` due to unexpected shape %s.",
                                name,
                                path,
                                tuple(tensor.shape),
                            )
                            continue
                        # prefer the discovered name for later comparisons
                        self.embedding_weight_names = [name] + [
                            existing for existing in self.embedding_weight_names if existing != name
                        ]
                        logger.info(
                            "Loaded embedding weight `%s` for modality `%s` from `%s` (shape=%s).",
                            name,
                            self.modality_id,
                            path,
                            tuple(tensor.shape),
                        )
                        return tensor
        logger.error(
            "Could not find any embedding weights %s in `%s` for modality `%s`.",
            self.embedding_weight_names,
            model_dir,
            self.modality_id,
        )
        return None

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
