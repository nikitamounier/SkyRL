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

from skyrl_train.modalities.checkpoint_utils import load_embedding_weight, load_module_checkpoint
from skyrl_train.modalities.handlers import ModalityEncoderProtocol, ModalityProjectorProtocol
from skyrl_train.utils.utils import str_to_torch_dtype

try:  # DTensor is optional depending on torch build
    from torch.distributed.tensor import DTensor, Replicate  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:  # older API
        from torch.distributed._tensor import DTensor, Replicate  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        DTensor = None  # type: ignore
        Replicate = None  # type: ignore


def _is_dtensor(value: torch.Tensor) -> bool:
    return DTensor is not None and isinstance(value, DTensor)


def _dtensor_mesh(value: torch.Tensor):
    if not _is_dtensor(value):
        return None
    mesh = getattr(value, "device_mesh", None) or getattr(value, "mesh", None)
    spec = getattr(value, "_spec", None)
    if mesh is None and spec is not None:
        mesh = getattr(spec, "mesh", None)
    return mesh


def _dtensor_to_local(value: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(value):
        return value
    if hasattr(value, "to_local"):
        return value.to_local()
    if hasattr(value, "local_tensor"):
        return value.local_tensor()
    return value


def _to_dtensor_replicated(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(ref):
        return tensor
    mesh = _dtensor_mesh(ref)
    if mesh is None or Replicate is None:
        return tensor
    # Use run_check=False to avoid collective syncs for small dummy inputs.
    return DTensor.from_local(tensor, mesh, [Replicate()], run_check=False)  # type: ignore[arg-type]


def _local_device_dtype(ref: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    local = _dtensor_to_local(ref)
    return local.device, local.dtype


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


def _import_memo_memory(memo_repo_root: Optional[str], pool_mode: str = "standard"):
    """Import and return the appropriate memory class based on pool_mode."""
    _maybe_add_memo_repo(memo_repo_root)
    try:
        from memo.models.memory import Memory, RefinedMemory, AdaptiveMemory, create_memory  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Failed to import `memo.models.memory`. "
            "Set MEMO_REPO_ROOT or pass memo_repo_root in handler kwargs."
        ) from exc
    _MEMORY_CLASSES = {
        "standard": Memory,
        "refined": RefinedMemory,
        "adaptive": AdaptiveMemory,
    }
    return _MEMORY_CLASSES.get(pool_mode, Memory)


def _import_memo_create_memory(memo_repo_root: Optional[str]):
    """Import the create_memory factory function."""
    _maybe_add_memo_repo(memo_repo_root)
    from memo.models.memory import create_memory  # type: ignore
    return create_memory


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


def _load_memory_checkpoint(
    memory_module: nn.Module,
    *,
    checkpoint_path: str,
    prefix: str,
    modality_id: str,
) -> int:
    missing, unexpected = load_module_checkpoint(
        memory_module,
        checkpoint_path=checkpoint_path,
        prefix=prefix,
        module_name=f"memory module `{modality_id}`",
    )
    total_params = len(memory_module.state_dict())
    loaded_params = total_params - len(missing)
    logger.info(
        "Checkpoint {} for modality `{}`: loaded {}/{} params (missing={}, unexpected={}).",
        checkpoint_path,
        modality_id,
        loaded_params,
        total_params,
        len(missing),
        len(unexpected),
    )
    return loaded_params


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
        pool_mode: str = "standard",
        num_cross_attn_layers: int = 2,
        projection_type: str = "linear",
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
        self.pool_mode = pool_mode

        create_memory = _import_memo_create_memory(memo_repo_root)
        self.memory = create_memory(
            pool_mode=pool_mode,
            embedding_dim=embedding_dim,
            num_memories=num_memories,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            memory_init=memory_init,
            num_cross_attn_layers=num_cross_attn_layers,
            projection_type=projection_type,
        )

        # Store for validation
        self.num_memories = num_memories
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self._target_device = torch.device(device) if device else None
        self._target_dtype = str_to_torch_dtype(dtype) if dtype else None
        if self._target_device or self._target_dtype:
            self.to(device=self._target_device, dtype=self._target_dtype)

        if checkpoint_path:
            _load_memory_checkpoint(
                self.memory,
                checkpoint_path=checkpoint_path,
                prefix=checkpoint_prefix,
                modality_id=self.modality_id,
            )

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
        ref_tensor = param if param is not None else None
        if ref_tensor is not None:
            device, dtype = _local_device_dtype(ref_tensor)
        else:
            device, dtype = None, None

        for payload in payloads:
            resolved = self._resolve_payload(payload)
            if resolved.precomputed:
                memory_embeddings = self._ensure_2d(resolved.inputs_embeds, name="memory_embeddings")
                memory_embeddings = _dtensor_to_local(memory_embeddings)
                outputs.append(memory_embeddings)
                continue

            inputs_embeds = _as_tensor(resolved.inputs_embeds, dtype=dtype, device=device)
            inputs_embeds = self._ensure_2d(inputs_embeds, name="inputs_embeds").unsqueeze(0)
            if ref_tensor is not None and _is_dtensor(ref_tensor):
                inputs_embeds = _to_dtensor_replicated(inputs_embeds, ref_tensor)

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
            memory_embeddings = _dtensor_to_local(memory_embeddings)
            outputs.append(self._ensure_2d(memory_embeddings, name="memory_embeddings"))

        return outputs


class MemoTokenMemoryEncoder(nn.Module, ModalityEncoderProtocol):
    """Memory encoder that accepts token ids and embeds them with frozen LLM embeddings."""

    DEFAULT_EMBEDDING_NAMES = (
        "model.embed_tokens.weight",
        "model.wte.weight",
    )

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        model_path: str,
        embedding_dim: int,
        num_memories: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        memory_init: str = "xavier_uniform",
        pool_mode: str = "standard",
        num_cross_attn_layers: int = 2,
        projection_type: str = "linear",
        embedding_weight_names: Optional[Sequence[str]] = None,
        max_doc_tokens: int = 256,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        memo_repo_root: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_prefix: str = "memory.",
        **_: Any,
    ) -> None:
        super().__init__()
        self.modality_id = modality_id
        self.role = role
        self.max_doc_tokens = int(max_doc_tokens)
        self.pool_mode = pool_mode

        create_memory = _import_memo_create_memory(memo_repo_root)
        self.memory = create_memory(
            pool_mode=pool_mode,
            embedding_dim=embedding_dim,
            num_memories=num_memories,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            memory_init=memory_init,
            num_cross_attn_layers=num_cross_attn_layers,
            projection_type=projection_type,
        )

        self.num_memories = num_memories
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        embedding_weight_names = list(embedding_weight_names or self.DEFAULT_EMBEDDING_NAMES)
        embedding_weight, _ = load_embedding_weight(
            model_path=model_path,
            embedding_weight_names=embedding_weight_names,
            modality_id=modality_id,
        )
        if embedding_weight.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch for `{modality_id}`: weight dim {embedding_weight.shape[1]} "
                f"!= expected {embedding_dim}."
            )

        target_dtype = next(self.memory.parameters(), embedding_weight).dtype
        if embedding_weight.dtype != target_dtype:
            embedding_weight = embedding_weight.to(dtype=target_dtype)
        # Store embedding weight as a buffer (not a parameter) to prevent FSDP from managing it
        # Buffers are not trainable and won't be included in optimizer or FSDP sharding
        self.register_buffer('embedding_weight', embedding_weight, persistent=True)
        self.embedding_dim_size = embedding_weight.shape[1]

        self._target_device = torch.device(device) if device else None
        self._target_dtype = str_to_torch_dtype(dtype) if dtype else None
        if self._target_device or self._target_dtype:
            self.to(device=self._target_device, dtype=self._target_dtype)

        if checkpoint_path:
            _load_memory_checkpoint(
                self.memory,
                checkpoint_path=checkpoint_path,
                prefix=checkpoint_prefix,
                modality_id=self.modality_id,
            )

    def encode(self, payloads: Sequence[Any]) -> Sequence[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        ref_param = next(self.memory.parameters(), None)
        ref_tensor = ref_param if ref_param is not None else self.embedding_weight
        device, dtype = _local_device_dtype(ref_tensor)

        embedding_dim = getattr(self, 'embedding_dim', getattr(self.memory, 'embedding_dim', 2560))
        num_memories = getattr(self, 'num_memories', getattr(self.memory, 'num_memories', 8))
        output_dim = getattr(self, 'output_dim', getattr(self.memory, 'output_dim', 2560))

        for idx, payload in enumerate(payloads):
            token_ids = self._resolve_token_ids(payload)
            if not token_ids:
                # No document content — return zeros. No gradient flows here,
                # which is correct: there is nothing for the Memory module to
                # learn from when no documents exist.  Gradient signal comes
                # from other samples/turns that do have real memory documents.
                outputs.append(torch.zeros(num_memories, output_dim, device=device, dtype=dtype))
                continue

            token_ids = token_ids[: self.max_doc_tokens]
            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
            embeddings = torch.nn.functional.embedding(token_tensor, self.embedding_weight)
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)

            # Detach to prevent gradients flowing to frozen LLM embedding weight
            embeddings_detached = embeddings.detach()

            # Ensure regular tensor (not DTensor) since Memory module is outside FSDP
            if _is_dtensor(embeddings_detached):
                embeddings_detached = _dtensor_to_local(embeddings_detached)

            if embeddings_detached.dim() != 3 or embeddings_detached.shape[0] != 1:
                raise RuntimeError(
                    f"Sample {idx}: expected shape [1, seq_len, {embedding_dim}], "
                    f"got {tuple(embeddings_detached.shape)}"
                )

            # Memory module forward pass — gradient flows through Memory params
            with torch.enable_grad():
                memory_embeddings, _ = self.memory(embeddings_detached, padding_mask=None)

            expected_shape = (1, num_memories, output_dim)
            if memory_embeddings.shape != expected_shape:
                raise RuntimeError(
                    f"Sample {idx}: Memory module returned {tuple(memory_embeddings.shape)}, "
                    f"expected {expected_shape}"
                )

            outputs.append(memory_embeddings.squeeze(0).to(dtype=dtype))

        return outputs

    def _resolve_token_ids(self, payload: Any) -> List[int]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            for key in ("token_ids", "input_ids", "tokens"):
                if key in payload:
                    return self._coerce_token_ids(payload[key])
        return self._coerce_token_ids(payload)

    def _coerce_token_ids(self, value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        raise TypeError(
            f"Modality `{self.modality_id}` expected token id list, got {type(value).__name__}."
        )


class MemoSentenceEmbeddingEncoder(nn.Module, ModalityEncoderProtocol):
    """Memory encoder using a dedicated embedding model (e.g. Qwen3-Embedding-4B).

    Matches the MeMo reference approach: documents are embedded with a
    SentenceTransformer model producing one semantic vector per document,
    then passed through the Memory module to produce memory tokens.

    Payloads should be raw text strings (not token ids).
    For adaptive mode, payloads should be dicts with "documents" and "observation" keys.
    """

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-4B",
        embedding_dim: int = 2560,
        num_memories: int = 8,
        output_dim: int = 2560,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        memory_init: str = "xavier_uniform",
        pool_mode: str = "standard",
        num_cross_attn_layers: int = 2,
        projection_type: str = "linear",
        embedding_device: str = "cpu",
        memo_repo_root: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_prefix: str = "memory.",
        **_: Any,
    ) -> None:
        super().__init__()
        self.modality_id = modality_id
        self.role = role
        self.embedding_dim = embedding_dim
        self.num_memories = num_memories
        self.output_dim = output_dim
        self.pool_mode = pool_mode

        # Load the Memory module via factory
        create_memory = _import_memo_create_memory(memo_repo_root)
        self.memory = create_memory(
            pool_mode=pool_mode,
            embedding_dim=embedding_dim,
            num_memories=num_memories,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            memory_init=memory_init,
            num_cross_attn_layers=num_cross_attn_layers,
            projection_type=projection_type,
        )

        # Small-scale init the output projection to prevent step-1 gradient
        # explosion. Pure zeros would make memory invisible (no grad_fn → no
        # backward). Small random values (~0.01 scale) produce small non-zero
        # outputs that create gradient signal without destabilizing the LLM.
        if hasattr(self.memory, 'memory_projection'):
            proj = self.memory.memory_projection
            # memory_projection can be nn.Linear or nn.Sequential (for MLP)
            if isinstance(proj, nn.Linear):
                nn.init.normal_(proj.weight, mean=0.0, std=0.01)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            elif isinstance(proj, nn.Sequential):
                # Init last linear layer in the MLP
                for layer in reversed(list(proj.children())):
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

        # Load embedding model on CPU to avoid GPU memory competition
        self._embedding_device = embedding_device
        self._embedding_model_name = embedding_model_name
        # IMPORTANT: hold the (frozen, lazily-loaded) SentenceTransformer inside a
        # plain list so that `nn.Module.__setattr__` does NOT register it as a
        # submodule. Otherwise, once it is lazy-loaded on the first `encode()`
        # call, its ~2B frozen params (`_embedding_model.0.auto_model.*`) leak
        # into this encoder's `state_dict()` and get written to checkpoints.
        # On resume, `load_checkpoint` runs before any forward pass, so the fresh
        # model has not lazy-loaded the embedder yet and its `state_dict()` lacks
        # those keys -> `load_state_dict(strict=True)` raises "Unexpected key(s)".
        # The embedding model is frozen and reloaded from the HF hub every run,
        # so it must never participate in checkpointing.
        self._embedding_model_holder: List[Any] = [None]  # Lazy-load on first use

        if checkpoint_path:
            loaded_params = _load_memory_checkpoint(
                self.memory,
                checkpoint_path=checkpoint_path,
                prefix=checkpoint_prefix,
                modality_id=self.modality_id,
            )
            if loaded_params == 0:
                logger.error(
                    "No parameters loaded from checkpoint `{}` for modality `{}`. "
                    "Check checkpoint_prefix (`{}`) and checkpoint key format.",
                    checkpoint_path,
                    self.modality_id,
                    checkpoint_prefix,
                )

    def _get_embedding_model(self):
        """Lazy-load the SentenceTransformer model on first use.

        The model is stored inside ``self._embedding_model_holder`` (a plain
        list) rather than as a direct attribute, so it is never registered as a
        submodule and never leaks into ``state_dict()`` / checkpoints.
        """
        if self._embedding_model_holder[0] is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading embedding model %s on %s",
                self._embedding_model_name,
                self._embedding_device,
            )
            self._embedding_model_holder[0] = SentenceTransformer(
                self._embedding_model_name,
                model_kwargs={
                    "device_map": self._embedding_device,
                    "torch_dtype": torch.bfloat16,
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
        return self._embedding_model_holder[0]

    def encode(self, payloads: Sequence[Any]) -> Sequence[torch.Tensor]:
        """Encode payloads into memory tokens (batched).

        Batch-embeds all docs in one SentenceTransformer call, then runs
        a single padded Memory module forward pass.

        For adaptive mode, uses mean-pooled document embeddings as the
        question signal (no separate observation needed — docs already
        contain the full context in TextWorld).
        """
        ref_param = next(self.memory.parameters(), None)
        device = ref_param.device if ref_param is not None else torch.device("cpu")
        dtype = ref_param.dtype if ref_param is not None else torch.bfloat16

        # Question-conditioned memory (AdaptiveMemory, AnchoredSelectorMemory,
        # InvertedAdaptiveMemory, CosineRAGMemory, RagAdaptive) takes a separate
        # question signal. Detect by type so any current/future variant works.
        try:
            from memo.models.memory import QUESTION_CONDITIONED_MEMORY_TYPES
            is_adaptive = isinstance(self.memory, QUESTION_CONDITIONED_MEMORY_TYPES)
        except Exception:
            is_adaptive = self.pool_mode == "adaptive"

        # Phase 1: resolve texts per payload
        all_texts: List[str] = []
        doc_counts: List[int] = []
        for payload in payloads:
            texts = self._resolve_texts(payload)
            doc_counts.append(len(texts))
            all_texts.extend(texts)

        # Phase 2: single batched SentenceTransformer encode
        all_embeddings = None
        if all_texts:
            model = self._get_embedding_model()
            with torch.no_grad():
                raw = model.encode(
                    all_texts, convert_to_numpy=False,
                    show_progress_bar=False, device=self._embedding_device,
                )
            if isinstance(raw, torch.Tensor):
                all_embeddings = raw.to(device=device, dtype=dtype)
            elif isinstance(raw, (list, tuple)):
                all_embeddings = torch.stack([
                    t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor)
                    else torch.tensor(t, device=device, dtype=dtype) for t in raw
                ])
            else:
                all_embeddings = torch.tensor(raw, device=device, dtype=dtype)

        # Phase 3: batched Memory forward with padding
        non_empty = [(i, dc) for i, dc in enumerate(doc_counts) if dc > 0]
        max_docs = max(doc_counts) if doc_counts else 0
        outputs: List[torch.Tensor] = [
            torch.zeros(self.num_memories, self.output_dim, device=device, dtype=dtype)
            for _ in payloads
        ]
        if not non_empty or max_docs == 0 or all_embeddings is None:
            return outputs

        B = len(non_empty)
        batch_embeds = torch.zeros(B, max_docs, self.embedding_dim, device=device, dtype=dtype)
        padding_mask = torch.ones(B, max_docs, dtype=torch.bool, device=device)
        offset = 0
        idx_map: List[int] = []
        for batch_pos, (orig_idx, dc) in enumerate(non_empty):
            batch_embeds[batch_pos, :dc] = all_embeddings[offset:offset + dc]
            padding_mask[batch_pos, :dc] = False
            offset += dc
            idx_map.append(orig_idx)

        with torch.enable_grad():
            if is_adaptive:
                # For AdaptiveMemory, use the last env observation as the
                # question signal — "what am I looking at now?" conditions
                # which past memory to surface.  If no observation text was
                # provided in the payload, fall back to mean-pooled docs.
                obs_texts_batch: List[str] = []
                for orig_idx in idx_map:
                    payload = payloads[orig_idx]
                    obs = ""
                    if isinstance(payload, dict):
                        obs = payload.get("observation", "")
                    obs_texts_batch.append(obs.strip() if isinstance(obs, str) else "")

                has_obs = any(obs_texts_batch)
                if has_obs:
                    model = self._get_embedding_model()
                    # Embed observations (use empty string fallback for missing)
                    obs_to_embed = [o if o else "no observation" for o in obs_texts_batch]
                    with torch.no_grad():
                        obs_raw = model.encode(
                            obs_to_embed, convert_to_numpy=False,
                            show_progress_bar=False, device=self._embedding_device,
                        )
                    if isinstance(obs_raw, torch.Tensor):
                        q_embeds = obs_raw.to(device=device, dtype=dtype)
                    else:
                        q_embeds = torch.stack([
                            t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor)
                            else torch.tensor(t, device=device, dtype=dtype) for t in obs_raw
                        ])
                else:
                    # Fallback: mean-pooled document embeddings
                    valid_mask = (~padding_mask).unsqueeze(-1).float()
                    q_embeds = (batch_embeds * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

                # Return arity varies by architecture (AdaptiveMemory -> 3-tuple
                # with gates; AnchoredSelectorMemory/others -> 2-tuple). Take [0].
                memory_out = self.memory(
                    batch_embeds.detach(),
                    q_embeds.detach(),
                    doc_padding_mask=padding_mask,
                )[0]
            else:
                memory_out = self.memory(batch_embeds.detach(), padding_mask=padding_mask)[0]

        for batch_pos, orig_idx in enumerate(idx_map):
            outputs[orig_idx] = memory_out[batch_pos].to(dtype=dtype)

        return outputs

    def _resolve_texts(self, payload: Any) -> List[str]:
        """Extract list of document texts from payload.

        Payload can be:
        - List[str]: list of document texts (expected from env)
        - dict: {"documents": [...], ...} (adaptive mode, extracts documents)
        - str: single document text
        - None/empty: no documents
        """
        if payload is None:
            return []
        if isinstance(payload, dict):
            docs = payload.get("documents", [])
            if isinstance(docs, (list, tuple)):
                return [t.strip() for t in docs if isinstance(t, str) and t.strip()]
            elif isinstance(docs, str) and docs.strip():
                return [docs.strip()]
            return []
        if isinstance(payload, str):
            text = payload.strip()
            return [text] if text else []
        if isinstance(payload, (list, tuple)):
            if payload and isinstance(payload[0], str):
                return [t.strip() for t in payload if isinstance(t, str) and t.strip()]
            if payload and isinstance(payload[0], int):
                logger.warning(
                    "MemoSentenceEmbeddingEncoder received token_ids instead of text. "
                    "Update TextWorld env to pass raw text payloads."
                )
                return []
        return []


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
    "MemoTokenMemoryEncoder",
    "MemoSentenceEmbeddingEncoder",
    "IdentityProjection",
    "LinearProjection",
]
