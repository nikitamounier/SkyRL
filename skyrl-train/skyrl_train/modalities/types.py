from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skyrl_train.dataset.modalities import ModalityEmbeddingSpan, ModalityPlaceholderPlan


@dataclass
class SampleModalityData:
    """All modality-specific information for a single sample."""

    payloads: Dict[str, Any] = field(default_factory=dict)
    plans: Dict[str, ModalityPlaceholderPlan] = field(default_factory=dict)
    embedding_spans: Dict[str, List[ModalityEmbeddingSpan]] = field(default_factory=dict)
    expanded_messages: Optional[List[Dict[str, Any]]] = None
    occurrence_payloads: Dict[str, List[Any]] = field(default_factory=dict)
    encoder_outputs: Dict[str, List[Any]] = field(default_factory=dict)
    projected_embeddings: Dict[str, List[Any]] = field(default_factory=dict)

    def clone(self) -> "SampleModalityData":
        """Deep copy to avoid sharing nested state across components."""
        return SampleModalityData(
            payloads=copy.deepcopy(self.payloads),
            plans=copy.deepcopy(self.plans),
            embedding_spans=copy.deepcopy(self.embedding_spans),
            expanded_messages=copy.deepcopy(self.expanded_messages),
            occurrence_payloads=copy.deepcopy(self.occurrence_payloads),
            encoder_outputs=copy.deepcopy(self.encoder_outputs),
            projected_embeddings=copy.deepcopy(self.projected_embeddings),
        )


ModalitiesMetadata = SampleModalityData


__all__ = ["SampleModalityData", "ModalitiesMetadata"]
