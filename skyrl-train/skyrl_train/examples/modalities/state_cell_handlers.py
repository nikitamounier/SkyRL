"""
Handlers for the BioReasonCell "STATE cell-embedding" modality.

A single precomputed STATE mean-embedding (2058-d) is attached to each sample and
injected at a ``<|cell_pad|>`` placeholder token, exactly mirroring BioReasonCell's
SFT/eval path (``CellProjectionMLP`` → scatter at the cell-pad position). The encoder
is a pure pass-through of the precomputed vector (no trainable params); the projector
is the 2-layer GELU MLP whose weights are loaded from the converted checkpoint's
``cell_projection.pt`` so the SkyRL forward is byte-identical to the SFT injection.

Registered via Hydra overrides, e.g.::

    +modalities.state_mod.placeholder_token="<|cell_pad|>"
    +modalities.state_mod.max_placeholder_tokens=1
    +modalities.state_mod.encoder.target=skyrl_train.examples.modalities.state_cell_handlers:StatePrecomputedEncoder
    +modalities.state_mod.projection.target=skyrl_train.examples.modalities.state_cell_handlers:CellProjection
    +modalities.state_mod.projection.kwargs.cell_projection_path=<ckpt>/cell_projection.pt
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
from loguru import logger

from skyrl_train.modalities.handlers import ModalityEncoderProtocol, ModalityProjectorProtocol

# BioReasonCell STATE mean-embedding dimensionality and the LLM hidden size the
# converted 9B checkpoint projects into. Both are overridable via kwargs.
STATE_EMBEDDING_DIM = 2058
LLM_HIDDEN_SIZE = 4096


class StatePrecomputedEncoder(ModalityEncoderProtocol):
    """Pass-through encoder for a single precomputed STATE mean-embedding per sample.

    Each payload is the raw 2058-d cell vector (a list/tuple of floats, a numpy
    array, or a tensor). It is returned as a 2D float tensor of shape
    ``(num_latents, embedding_dim)`` with ``num_latents == 1`` — one latent token,
    matching the single ``<|cell_pad|>`` slot the SFT model was trained with.
    """

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        embedding_dim: int = STATE_EMBEDDING_DIM,
        **_: Any,
    ) -> None:
        self.modality_id = modality_id
        self.role = role
        self.embedding_dim = embedding_dim

    def encode(self, payloads: Sequence[Any]) -> Sequence[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for idx, payload in enumerate(payloads):
            if isinstance(payload, torch.Tensor):
                tensor = payload.detach().to(dtype=torch.float32)
            else:
                tensor = torch.as_tensor(payload, dtype=torch.float32)
            # Accept (D,) or (1, D); normalise to (num_latents, D).
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() != 2 or tensor.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"Modality `{self.modality_id}` encoder expected a {self.embedding_dim}-d "
                    f"STATE vector; got shape {tuple(tensor.shape)} at index {idx}."
                )
            outputs.append(tensor)
        return outputs


class CellProjection(nn.Module, ModalityProjectorProtocol):
    """2-layer GELU MLP projecting a STATE vector (2058-d) to the LLM hidden size.

    Structurally identical to ``bioreason_cell.models.projection.CellProjectionMLP``
    (``Linear → GELU → Linear`` under a ``mlp`` Sequential), so the trained
    ``cell_projection.pt`` state dict loads directly. Set ``trainable.projection``
    to control whether these weights receive gradients during RL.
    """

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        model_path: Optional[str] = None,
        cell_projection_path: Optional[str] = None,
        embedding_dim: int = STATE_EMBEDDING_DIM,
        output_dim: int = LLM_HIDDEN_SIZE,
        dtype: str = "bfloat16",
        **_: Any,
    ) -> None:
        super().__init__()
        self.modality_id = modality_id
        self.role = role
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Resolve the trained projection weights: explicit path wins, else look
        # for ``cell_projection.pt`` next to the model checkpoint.
        resolved = cell_projection_path
        if resolved is None and model_path is not None:
            candidate = os.path.join(model_path, "cell_projection.pt")
            if os.path.isfile(candidate):
                resolved = candidate
        if resolved is not None:
            self._load_projection(resolved)
        else:
            logger.warning(
                "Modality `{}`: no cell_projection.pt found (model_path={}, "
                "cell_projection_path={}); using randomly initialised projection.",
                self.modality_id,
                model_path,
                cell_projection_path,
            )

        # Match the base model dtype (cell_projection.pt is stored fp32); FSDP
        # requires uniform dtype across flattened params, and BioReasonCell's
        # eval casts the projection to the model dtype too.
        self.to(self._dtype)

    def _load_projection(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        # The checkpoint stores exactly the ``mlp.*`` keys; anything else is a
        # structural mismatch worth surfacing loudly.
        if missing or unexpected:
            logger.warning(
                "Modality `{}`: cell_projection load had missing={} unexpected={}.",
                self.modality_id,
                list(missing),
                list(unexpected),
            )
        logger.info("Modality `{}`: loaded cell projection from {}.", self.modality_id, path)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        # features: (num_latents, embedding_dim) -> (num_latents, output_dim).
        # Match the projection's device+dtype: in the training forward the encoded
        # features arrive on CPU while the projection lives on the GPU.
        weight = self.mlp[0].weight
        return self.mlp(features.to(device=weight.device, dtype=weight.dtype))


__all__ = ["StatePrecomputedEncoder", "CellProjection"]
