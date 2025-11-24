from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from skyrl_train.dataset.modalities import ModalityPlaceholderPlan
from skyrl_train.modalities.types import SampleModalityData


@dataclass
class ModalityOccurrence:
    """Represents a single occurrence of a modality placeholder within a batch sample."""

    sample_index: int
    occurrence_index: int
    reserved_tokens: int
    payload: Any
    embedding_span: Optional[tuple[int, int]] = None  # (token_start, token_length)


@dataclass
class ModalityBatch:
    """Aggregated modality data across a batch of prompts."""

    modality_id: str
    occurrences: List[ModalityOccurrence] = field(default_factory=list)

    def add_occurrence(self, occurrence: ModalityOccurrence) -> None:
        self.occurrences.append(occurrence)

    @property
    def num_occurrences(self) -> int:
        return len(self.occurrences)


def _split_payload(plan: ModalityPlaceholderPlan) -> List[Any]:
    occurrences = plan.occurrences
    if occurrences == 0:
        return []

    payload = plan.payload
    if isinstance(payload, (list, tuple)):
        values = list(payload)
    else:
        values = [payload]

    if len(values) < occurrences:
        # pad missing payloads with None to keep alignment
        logger.warning(
            "Modality `%s` expected %d payload entries but received %d; padding missing occurrences with None.",
            plan.modality_id,
            occurrences,
            len(values),
        )
        values.extend([None] * (occurrences - len(values)))

    return values[:occurrences]


def populate_sample_occurrences(sample: SampleModalityData) -> None:
    """Ensure occurrence-level payload lists are computed for a sample."""
    if not sample.plans:
        return
    for modality_id, plan in sample.plans.items():
        sample.occurrence_payloads[modality_id] = _split_payload(plan)


def build_modality_batches(samples: Sequence[SampleModalityData]) -> Dict[str, ModalityBatch]:
    """Aggregate modality occurrences across samples to prepare encoder inputs."""
    modality_batches: Dict[str, ModalityBatch] = {}

    for sample_idx, sample in enumerate(samples):
        populate_sample_occurrences(sample)
        for modality_id, plan in sample.plans.items():
            payloads = sample.occurrence_payloads.get(modality_id, [])
            spans = sample.embedding_spans.get(modality_id, [])
            if plan.occurrences != len(payloads):
                logger.warning(
                    "Mismatch between occurrences and payload count for modality `%s`: %d vs %d.",
                    modality_id,
                    plan.occurrences,
                    len(payloads),
                )
            if modality_id not in modality_batches:
                modality_batches[modality_id] = ModalityBatch(modality_id=modality_id)
            for occurrence_idx in range(plan.occurrences):
                reserved_tokens = plan.reserved_tokens[occurrence_idx] if occurrence_idx < len(plan.reserved_tokens) else 0
                payload = payloads[occurrence_idx] if occurrence_idx < len(payloads) else None
                embedding_span = None
                if occurrence_idx < len(spans):
                    span = spans[occurrence_idx]
                    embedding_span = (span.token_start, span.token_length)
                occurrence = ModalityOccurrence(
                    sample_index=sample_idx,
                    occurrence_index=occurrence_idx,
                    reserved_tokens=reserved_tokens,
                    payload=payload,
                    embedding_span=embedding_span,
                )
                modality_batches[modality_id].add_occurrence(occurrence)

    return modality_batches


def subset_modality_batches(
    modality_batches: Optional[Dict[str, ModalityBatch]],
    selected_indices: Sequence[int],
) -> Dict[str, ModalityBatch]:
    if not modality_batches or not selected_indices:
        return {}

    index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(selected_indices)}
    subset: Dict[str, ModalityBatch] = {}

    for modality_id, batch in modality_batches.items():
        new_batch = ModalityBatch(modality_id=modality_id)
        for occurrence in batch.occurrences:
            if occurrence.sample_index not in index_map:
                continue
            new_batch.add_occurrence(
                ModalityOccurrence(
                    sample_index=index_map[occurrence.sample_index],
                    occurrence_index=occurrence.occurrence_index,
                    reserved_tokens=occurrence.reserved_tokens,
                    payload=occurrence.payload,
                    embedding_span=occurrence.embedding_span,
                )
            )
        if new_batch.occurrences:
            subset[modality_id] = new_batch

    return subset


def subset_modalities_metadata(
    metadata_list: Optional[List[SampleModalityData]],
    selected_indices: Sequence[int],
) -> Optional[List[SampleModalityData]]:
    if metadata_list is None:
        return None
    return [metadata_list[i] for i in selected_indices]


__all__ = [
    "ModalityOccurrence",
    "ModalityBatch",
    "populate_sample_occurrences",
    "build_modality_batches",
    "subset_modalities_metadata",
]
