from __future__ import annotations

from typing import Protocol, runtime_checkable, Sequence, Any, List
import torch


@runtime_checkable
class ModalityEncoderProtocol(Protocol):
    """Minimal interface expected from modality encoders."""

    def encode(self, payloads: Sequence[Any]) -> Sequence[torch.Tensor]:
        ...


@runtime_checkable
class ModalityProjectorProtocol(Protocol):
    """Minimal interface expected from modality projection modules."""

    def project(self, features: torch.Tensor) -> torch.Tensor:
        ...


def ensure_sequence_of_tensors(
    values: Sequence[Any],
    *,
    name: str,
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for idx, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} expected a sequence of tensors; got {type(value).__name__} at index {idx}.")
        tensors.append(value)
    return tensors


__all__ = [
    "ModalityEncoderProtocol",
    "ModalityProjectorProtocol",
    "ensure_sequence_of_tensors",
]
