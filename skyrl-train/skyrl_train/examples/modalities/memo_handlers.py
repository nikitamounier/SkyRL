"""
MeMo modality handlers for SkyRL.

This module adapts the MeMo memory module so it can be used as a modality encoder.
It expects each modality payload to provide document embeddings (or a user_id that
maps to preloaded embeddings). The memory module produces k memory vectors which
are then injected into the prompt via modality placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from skyrl_train.modalities.handlers import ModalityEncoderProtocol, ModalityProjectorProtocol
from skyrl_train.utils.utils import str_to_torch_dtype


def _maybe_add_memo_repo(memo_repo_root: Optional[str]) -> None:
    if importlib.util.find_spec("memo") is not None:
        return

    candidates: List[Path] = []
    if memo_repo_root:
        candidates.append(Path(memo_repo_root))
    env_path = os.environ.get("MEMO_REPO_ROOT")
    if env_path:
        candidates.append(Path(env_path))

    try:
        # memo_handlers.py -> modalities -> examples -> skyrl_train -> skyrl-train -> <repo_root>
        repo_root = Path(__file__).resolve().parents[4].parent
        candidates.append(repo_root.parent / "MeMo")
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_dir():
            sys_path = str(candidate)
            if sys_path not in sys.path:
                sys.path.append(sys_path)
            if importlib.util.find_spec("memo") is not None:
                return


def _import_memo_memory(memo_repo_root: Optional[str]):
    _maybe_add_memo_repo(memo_repo_root)
    try:
        from memo.models.memory import Memory  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Failed to import `memo.models.memory.Memory`. "
            "Set MEMO_REPO_ROOT or pass memo_repo_root in handler kwargs."
        ) from exc
    return Memory


def _import_memo_embed(memo_repo_root: Optional[str]):
    _maybe_add_memo_repo(memo_repo_root)
    try:
        from memo.dataset.embed import load_document_embeddings  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Failed to import `memo.dataset.embed.load_document_embeddings`. "
            "Set MEMO_REPO_ROOT or pass memo_repo_root in handler kwargs."
        ) from exc
    return load_document_embeddings


def _as_tensor(value: Any, *, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor


@dataclass
class MemoryPayload:
    inputs_embeds: torch.Tensor
    padding_mask: Optional[torch.Tensor]
    precomputed: bool = False


class MemoryEmbeddingsBank:
    """Holds per-user document embeddings on CPU (or a specified device)."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings to be 3D, got shape {tuple(embeddings.shape)}.")
        if padding_mask.dim() != 2:
            raise ValueError(f"Expected padding_mask to be 2D, got shape {tuple(padding_mask.shape)}.")
        if embeddings.shape[:2] != padding_mask.shape:
            raise ValueError(
                "Embeddings and padding_mask shapes are incompatible: "
                f"{tuple(embeddings.shape)} vs {tuple(padding_mask.shape)}."
            )
        if dtype is not None:
            embeddings = embeddings.to(dtype=dtype)
        if device is not None:
            embeddings = embeddings.to(device=device)
            padding_mask = padding_mask.to(device=device)
        self.embeddings = embeddings
        self.padding_mask = padding_mask

    @property
    def num_users(self) -> int:
        return self.embeddings.shape[0]

    @classmethod
    def from_embeddings_files(
        cls,
        embeddings_files: Sequence[str],
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "MemoryEmbeddingsBank":
        tensors: List[torch.Tensor] = []
        lengths: List[int] = []
        for path in embeddings_files:
            array = np.load(path)
            tensor = torch.from_numpy(array)
            tensors.append(tensor)
            lengths.append(tensor.shape[0])
        padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        max_docs = padded.shape[1]
        lengths_tensor = torch.tensor(lengths)
        padding_mask = torch.arange(max_docs).unsqueeze(0) >= lengths_tensor.unsqueeze(1)
        return cls(padded, padding_mask, device=device, dtype=dtype)

    @classmethod
    def from_memo_documents(
        cls,
        memory_documents_dirs: Sequence[str],
        memory_embeddings_files: Sequence[str],
        *,
        memo_repo_root: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> "MemoryEmbeddingsBank":
        load_document_embeddings = _import_memo_embed(memo_repo_root)
        embeddings, padding_mask = load_document_embeddings(
            memory_documents_dirs=list(memory_documents_dirs),
            memory_embeddings_files=list(memory_embeddings_files),
            **kwargs,
        )
        return cls(embeddings, padding_mask, device=device, dtype=dtype)

    def get(self, user_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if user_id < 0 or user_id >= self.num_users:
            raise IndexError(f"user_id {user_id} is out of range for {self.num_users} users.")
        return self.embeddings[user_id], self.padding_mask[user_id]


class MemoMemoryEncoder(nn.Module, ModalityEncoderProtocol):
    """Wraps the MeMo Memory module to serve as a modality encoder."""

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        embedding_dim: int,
        num_memories: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        memory_init: str = "xavier_uniform",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        memo_repo_root: Optional[str] = None,
        memory_embeddings_files: Optional[Sequence[str]] = None,
        memory_documents_dirs: Optional[Sequence[str]] = None,
        memory_bank: Optional[MemoryEmbeddingsBank] = None,
        memory_bank_device: Optional[str] = None,
        memory_bank_dtype: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_prefix: str = "memory.",
        **_: Any,
    ) -> None:
        super().__init__()
        self.modality_id = modality_id
        self.role = role

        Memory = _import_memo_memory(memo_repo_root)
        self.memory = Memory(
            embedding_dim=embedding_dim,
            num_memories=num_memories,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            memory_init=memory_init,
        )

        self._target_device = torch.device(device) if device else None
        self._target_dtype = str_to_torch_dtype(dtype) if dtype else None
        if self._target_device or self._target_dtype:
            self.to(device=self._target_device, dtype=self._target_dtype)

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, prefix=checkpoint_prefix)

        self.memory_bank = memory_bank
        if self.memory_bank is None and memory_embeddings_files:
            bank_dtype = str_to_torch_dtype(memory_bank_dtype) if memory_bank_dtype else None
            bank_device = torch.device(memory_bank_device) if memory_bank_device else None
            self.memory_bank = MemoryEmbeddingsBank.from_embeddings_files(
                memory_embeddings_files,
                dtype=bank_dtype,
                device=bank_device,
            )
        if self.memory_bank is None and memory_documents_dirs and memory_embeddings_files:
            bank_dtype = str_to_torch_dtype(memory_bank_dtype) if memory_bank_dtype else None
            bank_device = torch.device(memory_bank_device) if memory_bank_device else None
            self.memory_bank = MemoryEmbeddingsBank.from_memo_documents(
                memory_documents_dirs=memory_documents_dirs,
                memory_embeddings_files=memory_embeddings_files,
                memo_repo_root=memo_repo_root,
                dtype=bank_dtype,
                device=bank_device,
            )

    def _load_checkpoint(self, checkpoint_path: str, *, prefix: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a state_dict.")
        filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not filtered:
            logger.warning(
                "No parameters matched prefix '%s' in checkpoint %s; loading full state_dict.",
                prefix,
                checkpoint_path,
            )
            filtered = state_dict
        missing, unexpected = self.memory.load_state_dict(filtered, strict=False)
        if missing:
            logger.warning("Missing keys when loading memory checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading memory checkpoint: %s", unexpected)

    def _resolve_payload(self, payload: Any) -> MemoryPayload:
        if payload is None:
            raise ValueError(f"Modality `{self.modality_id}` received a null payload.")

        # User id lookup
        if isinstance(payload, (int, np.integer)):
            if self.memory_bank is None:
                raise ValueError(
                    f"Modality `{self.modality_id}` received user_id but no memory_bank was configured."
                )
            inputs_embeds, padding_mask = self.memory_bank.get(int(payload))
            return MemoryPayload(inputs_embeds=inputs_embeds, padding_mask=padding_mask)

        # Dict payload
        if isinstance(payload, dict):
            if "user_id" in payload:
                if self.memory_bank is None:
                    raise ValueError(
                        f"Modality `{self.modality_id}` received user_id but no memory_bank was configured."
                    )
                inputs_embeds, padding_mask = self.memory_bank.get(int(payload["user_id"]))
                return MemoryPayload(inputs_embeds=inputs_embeds, padding_mask=padding_mask)

            if "memory_embeddings" in payload:
                return MemoryPayload(
                    inputs_embeds=_as_tensor(payload["memory_embeddings"]),
                    padding_mask=None,
                    precomputed=True,
                )

            inputs_embeds = (
                payload.get("memory_inputs_embeds")
                or payload.get("inputs_embeds")
                or payload.get("embeddings")
            )
            padding_mask = payload.get("memory_padding_mask") or payload.get("padding_mask")
            attention_mask = payload.get("attention_mask")
            if inputs_embeds is not None:
                if padding_mask is None and attention_mask is not None:
                    padding_mask = ~_as_tensor(attention_mask).to(dtype=torch.bool)
                return MemoryPayload(
                    inputs_embeds=_as_tensor(inputs_embeds),
                    padding_mask=_as_tensor(padding_mask) if padding_mask is not None else None,
                )

        # Tuple/list payload
        if isinstance(payload, (list, tuple)) and payload:
            inputs_embeds = payload[0]
            padding_mask = payload[1] if len(payload) > 1 else None
            return MemoryPayload(
                inputs_embeds=_as_tensor(inputs_embeds),
                padding_mask=_as_tensor(padding_mask) if padding_mask is not None else None,
            )

        # Raw tensor payload
        if isinstance(payload, torch.Tensor):
            return MemoryPayload(inputs_embeds=payload, padding_mask=None)

        raise TypeError(
            f"Unsupported payload type for modality `{self.modality_id}`: {type(payload).__name__}."
        )

    def _ensure_2d(self, tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 3 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        raise ValueError(f"{name} must be 2D (or batch=1), got shape {tuple(tensor.shape)}.")

    def encode(self, payloads: Sequence[Any]) -> Sequence[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        param = next(self.memory.parameters(), None)
        device = param.device if param is not None else None
        dtype = param.dtype if param is not None else None

        for payload in payloads:
            resolved = self._resolve_payload(payload)
            if resolved.precomputed:
                memory_embeddings = self._ensure_2d(resolved.inputs_embeds, name="memory_embeddings")
                outputs.append(memory_embeddings)
                continue

            inputs_embeds = _as_tensor(resolved.inputs_embeds, dtype=dtype, device=device)
            inputs_embeds = self._ensure_2d(inputs_embeds, name="inputs_embeds").unsqueeze(0)

            padding_mask = resolved.padding_mask
            if padding_mask is not None:
                padding_mask = _as_tensor(padding_mask, device=device)
                if padding_mask.dim() == 1:
                    padding_mask = padding_mask.unsqueeze(0)
                elif padding_mask.dim() == 3 and padding_mask.size(0) == 1:
                    padding_mask = padding_mask.squeeze(0)
                if padding_mask.dim() != 2:
                    raise ValueError(
                        f"padding_mask must be 1D or 2D (or batch=1), got shape {tuple(padding_mask.shape)}."
                    )
                padding_mask = padding_mask.to(dtype=torch.bool)

            memory_embeddings, _ = self.memory(inputs_embeds, padding_mask)
            outputs.append(self._ensure_2d(memory_embeddings, name="memory_embeddings"))

        return outputs


class IdentityProjection(nn.Module, ModalityProjectorProtocol):
    """Pass-through projection for memory embeddings."""

    def __init__(self, modality_id: str, role: str, **_: Any) -> None:
        super().__init__()
        self.modality_id = modality_id
        self.role = role

    def project(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(
                f"Projection for `{self.modality_id}` expected 2D tensor, got shape {tuple(features.shape)}."
            )
        return features


class LinearProjection(nn.Module, ModalityProjectorProtocol):
    """Linear projection to match the LLM hidden size if needed."""

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modality_id = modality_id
        self.role = role
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(
                f"Projection for `{self.modality_id}` expected 2D tensor, got shape {tuple(features.shape)}."
            )
        return self.proj(features)


__all__ = [
    "MemoryEmbeddingsBank",
    "MemoMemoryEncoder",
    "IdentityProjection",
    "LinearProjection",
]
