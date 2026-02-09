from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from loguru import logger

from skyrl_train.dataset.modalities import ModalitySpec, normalize_modalities_config
from skyrl_train.modalities.batching import ModalityBatch, ModalityOccurrence
from skyrl_train.modalities.handlers import (
    ModalityEncoderProtocol,
    ModalityProjectorProtocol,
    ensure_sequence_of_tensors,
)
from skyrl_train.modalities.loader import instantiate_handler
from skyrl_train.modalities.types import SampleModalityData


@dataclass
class _HandlerBundle:
    spec: ModalitySpec
    encoder: ModalityEncoderProtocol
    projector: ModalityProjectorProtocol


class ModalitiesManager:
    """Runtime helper that materializes modality embeddings for prompts."""

    def __init__(self, modality_specs: Mapping[str, ModalitySpec] | None):
        specs = modality_specs or {}
        if specs and not all(isinstance(spec, ModalitySpec) for spec in specs.values()):
            specs = normalize_modalities_config(specs)
        self._specs: Dict[str, ModalitySpec] = dict(specs)
        self._handlers: Dict[str, _HandlerBundle] = {}

        for modality_id, spec in self._specs.items():
            encoder = instantiate_handler(spec.encoder, modality_id=modality_id, role="encoder")
            projector = instantiate_handler(spec.projection, modality_id=modality_id, role="projection")

            self._configure_module_trainability(encoder, spec.trainable.encoder)
            self._configure_module_trainability(projector, spec.trainable.projection)

            self._handlers[modality_id] = _HandlerBundle(spec=spec, encoder=encoder, projector=projector)

        self._warn_if_empty()

    def _warn_if_empty(self):
        if not self._handlers:
            logger.debug("ModalitiesManager initialized with no modalities. Runtime will be a no-op.")

    @staticmethod
    def _configure_module_trainability(module: object, trainable: bool) -> None:
        if hasattr(module, "train") and hasattr(module, "eval"):
            if trainable:
                module.train()
            else:
                module.eval()
        if hasattr(module, "parameters"):
            for param in module.parameters():  # type: ignore[attr-defined]
                param.requires_grad = trainable

    def has_modalities(self) -> bool:
        return bool(self._handlers)

    def set_placeholder_token_ids(self, token_ids: Dict[str, int]) -> None:
        """Store resolved placeholder token IDs (modality_id -> token_id)."""
        self._placeholder_token_ids = dict(token_ids)

    def resolve_placeholder_token_ids(self, tokenizer) -> None:
        """Resolve placeholder token strings to token IDs using the given tokenizer."""
        self._placeholder_token_ids = {}
        for modality_id, spec in self._specs.items():
            tok_id = tokenizer.convert_tokens_to_ids(spec.placeholder_token)
            if tok_id is not None and tok_id != -1:
                self._placeholder_token_ids[modality_id] = tok_id
                logger.info(f"Resolved placeholder for '{modality_id}': '{spec.placeholder_token}' -> {tok_id}")

    def ensure_embedding_spans(
        self,
        input_ids: torch.LongTensor,
        samples_metadata: Sequence[SampleModalityData],
    ) -> None:
        """Compute missing embedding_spans from input_ids for samples that have plans but no spans.

        If the plan expects more occurrences than there are placeholder tokens in input_ids,
        the plan is trimmed to match what's actually available.
        """
        from skyrl_train.dataset.modalities import ModalityEmbeddingSpan, collect_embedding_spans

        token_ids_map = getattr(self, '_placeholder_token_ids', {})
        if not token_ids_map:
            return

        for sample_idx, meta in enumerate(samples_metadata):
            for mod_id, plan in list(meta.plans.items()):
                if plan.occurrences <= 0:
                    continue
                existing_spans = meta.embedding_spans.get(mod_id, [])
                if len(existing_spans) >= plan.occurrences:
                    continue
                tok_id = token_ids_map.get(mod_id)
                if tok_id is None:
                    continue
                sample_ids = input_ids[sample_idx].tolist()
                num_placeholder = sample_ids.count(tok_id)
                tokens_per_occ = plan.reserved_tokens[0] if plan.reserved_tokens else 0
                if tokens_per_occ <= 0:
                    continue

                # How many occurrences can the actual input_ids support?
                actual_occurrences = num_placeholder // tokens_per_occ
                if actual_occurrences <= 0:
                    # No placeholder tokens at all - remove the plan
                    del meta.plans[mod_id]
                    meta.payloads.pop(mod_id, None)
                    continue

                if actual_occurrences < plan.occurrences:
                    # Trim plan to match available tokens
                    plan.occurrences = actual_occurrences
                    plan.reserved_tokens = plan.reserved_tokens[:actual_occurrences]
                    if isinstance(plan.payload, (list, tuple)):
                        plan.payload = plan.payload[:actual_occurrences]

                try:
                    spans = collect_embedding_spans(sample_ids, tok_id, plan.reserved_tokens)
                    meta.embedding_spans[mod_id] = [
                        ModalityEmbeddingSpan(
                            modality_id=mod_id,
                            occurrence_index=i,
                            token_start=start,
                            token_length=length,
                        )
                        for i, (start, length) in enumerate(spans)
                    ]
                except ValueError as e:
                    logger.debug(f"ensure_embedding_spans: sample {sample_idx}, {mod_id}: {e}")
                    # Last resort: remove the plan entirely
                    del meta.plans[mod_id]
                    meta.payloads.pop(mod_id, None)

    def compute_embeddings(
        self,
        modality_batches: Mapping[str, ModalityBatch],
        samples_metadata: Sequence[SampleModalityData],
        *,
        target_device: torch.device,
        target_dtype: torch.dtype,
        non_blocking: bool = False,
        update_metadata: bool = True,
    ) -> Dict[int, List[Tuple[Tuple[int, int], torch.Tensor]]]:
        """Return per-sample embedding replacements and update metadata.

        Returns:
            Dict mapping sample index -> list of ``((token_start, length), tensor)``.
        """
        if not modality_batches:
            return {}

        replacements: Dict[int, List[Tuple[Tuple[int, int], torch.Tensor]]] = defaultdict(list)

        for modality_id, batch in modality_batches.items():
            if modality_id not in self._handlers:
                raise ValueError(f"Received modality `{modality_id}` which is not configured.")
            handler_bundle = self._handlers[modality_id]
            occurrences = batch.occurrences
            if not occurrences:
                continue

            payloads = [occ.payload for occ in occurrences]
            raw_features = self._run_encoder(handler_bundle, payloads)

            if len(raw_features) != len(occurrences):
                raise ValueError(
                    f"Encoder for modality `{modality_id}` returned {len(raw_features)} tensors for "
                    f"{len(occurrences)} occurrences."
                )

            for occ, raw in zip(occurrences, raw_features):
                span = occ.embedding_span
                if span is None:
                    raise ValueError(
                        f"Missing embedding span for modality `{modality_id}` occurrence "
                        f"(sample={occ.sample_index}, occurrence={occ.occurrence_index})."
                    )

                projected = self._run_projector(handler_bundle, raw)

                if projected.dim() != 2:
                    raise ValueError(
                        f"Projection for modality `{modality_id}` must return 2D tensor; got shape {projected.shape}."
                    )

                reserved_tokens = occ.reserved_tokens
                if reserved_tokens <= 0:
                    reserved_tokens = projected.shape[0]
                if projected.shape[0] != reserved_tokens:
                    raise ValueError(
                        f"Projected embedding length mismatch for modality `{modality_id}` occurrence "
                        f"(sample={occ.sample_index}, occurrence={occ.occurrence_index}). "
                        f"Expected {reserved_tokens} tokens, got {projected.shape[0]}."
                    )

                final_tensor = projected.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
                replacements[occ.sample_index].append(((span[0], span[1]), final_tensor))

                if update_metadata:
                    self._update_sample_metadata(
                        samples_metadata,
                        modality_id=modality_id,
                        occurrence=occ,
                        encoder_output=raw,
                        projected_output=projected,
                    )

        return replacements

    def _run_encoder(
        self,
        handler_bundle: _HandlerBundle,
        payloads: Sequence[object],
    ) -> List[torch.Tensor]:
        encoder = handler_bundle.encoder
        if handler_bundle.spec.trainable.encoder:
            outputs = encoder.encode(payloads)  # type: ignore[func-returns-value]
        else:
            with torch.no_grad():
                outputs = encoder.encode(payloads)  # type: ignore[func-returns-value]
        tensor_outputs = ensure_sequence_of_tensors(outputs, name=f"{handler_bundle.spec.modality_id} encoder output")
        if handler_bundle.spec.trainable.encoder:
            return list(tensor_outputs)
        return [tensor.detach() for tensor in tensor_outputs]

    def _run_projector(
        self,
        handler_bundle: _HandlerBundle,
        features: torch.Tensor,
    ) -> torch.Tensor:
        projector = handler_bundle.projector
        if features.requires_grad and not handler_bundle.spec.trainable.projection:
            features = features.detach()
        result = projector.project(features)  # type: ignore[func-returns-value]
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f"Projection module for modality `{handler_bundle.spec.modality_id}` must return a tensor, "
                f"got {type(result).__name__}."
            )
        return result

    @staticmethod
    def _update_sample_metadata(
        samples_metadata: Sequence[SampleModalityData],
        *,
        modality_id: str,
        occurrence: ModalityOccurrence,
        encoder_output: torch.Tensor,
        projected_output: torch.Tensor,
    ) -> None:
        if occurrence.sample_index >= len(samples_metadata):
            logger.warning(
                "Sample index %d for modality `%s` is out of bounds for metadata list of size %d.",
                occurrence.sample_index,
                modality_id,
                len(samples_metadata),
            )
            return

        sample_metadata = samples_metadata[occurrence.sample_index]
        encoded_store = sample_metadata.encoder_outputs.setdefault(modality_id, [])
        projected_store = sample_metadata.projected_embeddings.setdefault(modality_id, [])

        ModalitiesManager._ensure_list_size(encoded_store, occurrence.occurrence_index + 1)
        ModalitiesManager._ensure_list_size(projected_store, occurrence.occurrence_index + 1)

        encoded_store[occurrence.occurrence_index] = encoder_output.detach().cpu()
        projected_store[occurrence.occurrence_index] = projected_output.detach().cpu()

    @staticmethod
    def _ensure_list_size(store: List[object], size: int) -> None:
        if len(store) < size:
            store.extend([None] * (size - len(store)))

    def iter_handler_modules(self):
        """Yield (modality_id, role, module) tuples for modules managed by the manager."""
        for modality_id, bundle in self._handlers.items():
            if isinstance(bundle.encoder, nn.Module):
                yield modality_id, "encoder", bundle.encoder
            if isinstance(bundle.projector, nn.Module):
                yield modality_id, "projection", bundle.projector

    def get_module_by_name(self, name: str) -> Optional[nn.Module]:
        for modality_id, bundle in self._handlers.items():
            if name == f"{modality_id}.encoder" and isinstance(bundle.encoder, nn.Module):
                return bundle.encoder
            if name == f"{modality_id}.projection" and isinstance(bundle.projector, nn.Module):
                return bundle.projector
        return None

    def set_named_parameter(self, full_name: str, tensor: torch.Tensor) -> bool:
        if not full_name.startswith("modalities."):
            return False
        parts = full_name.split(".")
        if len(parts) < 4:
            logger.warning("Invalid modality parameter name: {}", full_name)
            return False
        modality_id, role = parts[1], parts[2]
        param_path = ".".join(parts[3:])
        bundle = self._handlers.get(modality_id)
        if bundle is None:
            logger.warning("No modality handler registered for `%s`", modality_id)
            return False
        module: Optional[nn.Module] = None
        if role == "encoder" and isinstance(bundle.encoder, nn.Module):
            module = bundle.encoder
        elif role == "projection" and isinstance(bundle.projector, nn.Module):
            module = bundle.projector
        else:
            logger.warning("Unknown modality role `%s` for modality `%s`", role, modality_id)
            return False

        for name, param in module.named_parameters():
            if name == param_path:
                tensor = tensor.to(param.dtype).to(param.device)
                param.data.copy_(tensor)
                return True

        logger.warning(
            "Parameter `%s` not found in modality `%s` role `%s`", param_path, modality_id, role
        )
        return False


__all__ = ["ModalitiesManager"]
