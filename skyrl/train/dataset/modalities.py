from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from loguru import logger


@dataclass(frozen=True)
class ModalityHandlerSpec:
    target: str
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class ModalityTrainableSpec:
    encoder: bool
    projection: bool


@dataclass(frozen=True)
class ModalitySpec:
    modality_id: str
    placeholder_token: str
    max_placeholder_tokens: int
    encoder: ModalityHandlerSpec
    projection: ModalityHandlerSpec
    trainable: ModalityTrainableSpec


@dataclass
class ModalityPlaceholderPlan:
    modality_id: str
    placeholder_token: str
    occurrences: int
    reserved_tokens: List[int]
    payload: Any = field(repr=False)


@dataclass
class ModalityEmbeddingSpan:
    modality_id: str
    occurrence_index: int
    token_start: int
    token_length: int


def _as_handler(handler_cfg: Mapping[str, Any], *, modality_id: str, handler_type: str) -> ModalityHandlerSpec:
    if "target" not in handler_cfg or not handler_cfg["target"]:
        raise ValueError(f"Modality `{modality_id}` is missing `{handler_type}.target` in configuration.")
    kwargs = handler_cfg.get("kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise ValueError(
            f"Modality `{modality_id}` `{handler_type}.kwargs` must be a mapping, "
            f"got {type(kwargs).__name__}."
        )
    return ModalityHandlerSpec(target=str(handler_cfg["target"]), kwargs=dict(kwargs))


def _as_trainable(cfg: Optional[Mapping[str, Any]], *, modality_id: str) -> ModalityTrainableSpec:
    cfg = cfg or {}
    encoder_trainable = bool(cfg.get("encoder", False))
    projection_trainable = bool(cfg.get("projection", True))
    return ModalityTrainableSpec(encoder=encoder_trainable, projection=projection_trainable)


def normalize_modalities_config(raw_config: Optional[Mapping[str, Any]]) -> Dict[str, ModalitySpec]:
    if not raw_config:
        return {}

    normalized: Dict[str, ModalitySpec] = {}
    for modality_id, modality_cfg in raw_config.items():
        if not isinstance(modality_cfg, Mapping):
            raise ValueError(
                f"Configuration for modality `{modality_id}` must be a mapping, "
                f"got {type(modality_cfg).__name__}."
            )

        placeholder_token = modality_cfg.get("placeholder_token")
        if not placeholder_token or not isinstance(placeholder_token, str):
            raise ValueError(f"Modality `{modality_id}` must define a string `placeholder_token`.")

        max_placeholder_tokens = modality_cfg.get("max_placeholder_tokens", 0)
        if not isinstance(max_placeholder_tokens, int) or max_placeholder_tokens < 0:
            raise ValueError(
                f"Modality `{modality_id}` `max_placeholder_tokens` must be a non-negative integer; "
                f"got {max_placeholder_tokens!r}."
            )

        try:
            encoder_spec = _as_handler(modality_cfg.get("encoder", {}), modality_id=modality_id, handler_type="encoder")
        except ValueError as exc:
            raise ValueError(f"Invalid encoder configuration for modality `{modality_id}`: {exc}") from exc

        try:
            projection_spec = _as_handler(
                modality_cfg.get("projection", {}), modality_id=modality_id, handler_type="projection"
            )
        except ValueError as exc:
            raise ValueError(f"Invalid projection configuration for modality `{modality_id}`: {exc}") from exc

        trainable_spec = _as_trainable(modality_cfg.get("trainable"), modality_id=modality_id)

        normalized[modality_id] = ModalitySpec(
            modality_id=modality_id,
            placeholder_token=placeholder_token,
            max_placeholder_tokens=max_placeholder_tokens,
            encoder=encoder_spec,
            projection=projection_spec,
            trainable=trainable_spec,
        )
    return normalized


def build_placeholder_lookup(specs: Mapping[str, ModalitySpec]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for modality_id, spec in specs.items():
        token = spec.placeholder_token
        if token in lookup and lookup[token] != modality_id:
            raise ValueError(
                f"Placeholder token `{token}` is shared by both `{lookup[token]}` and `{modality_id}` modalities."
            )
        lookup[token] = modality_id
    return lookup


def _count_placeholder_occurrences(messages: Sequence[Mapping[str, Any]], placeholder_token: str) -> int:
    count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            count += content.count(placeholder_token)
    return count


def _estimate_payload_lengths(payload: Any, occurrences: int) -> List[int]:
    if occurrences == 0:
        return []

    if isinstance(payload, (list, tuple)):
        values: Iterable[Any] = payload
    else:
        values = [payload]

    values_list = list(values)
    lengths: List[int] = []
    for idx in range(occurrences):
        candidate = values_list[idx] if idx < len(values_list) else None
        if isinstance(candidate, str):
            lengths.append(len(candidate))
        elif hasattr(candidate, "__len__"):
            try:
                lengths.append(len(candidate))  # type: ignore[arg-type]
            except TypeError:
                lengths.append(0)
        else:
            lengths.append(0)
    if len(lengths) < occurrences:
        lengths.extend([0] * (occurrences - len(lengths)))
    return lengths


def _resolve_reserved_tokens(spec: ModalitySpec, occurrences: int, payload: Any) -> List[int]:
    if occurrences == 0:
        return []

    estimated_lengths = _estimate_payload_lengths(payload, occurrences)
    reserved: List[int] = []
    for idx in range(occurrences):
        est = estimated_lengths[idx] if idx < len(estimated_lengths) else 0
        if spec.max_placeholder_tokens > 0:
            reserved_tokens = min(spec.max_placeholder_tokens, est) if est > 0 else spec.max_placeholder_tokens
        else:
            reserved_tokens = est if est > 0 else 1
        reserved.append(max(1, reserved_tokens))
    return reserved


def plan_modalities_for_prompt(
    messages: Sequence[Mapping[str, Any]],
    modalities_payload: Mapping[str, Any],
    specs: Mapping[str, ModalitySpec],
) -> Dict[str, ModalityPlaceholderPlan]:
    plans: Dict[str, ModalityPlaceholderPlan] = {}
    if not specs:
        return plans

    placeholder_lookup = build_placeholder_lookup(specs)

    for modality_id, spec in specs.items():
        occurrences = _count_placeholder_occurrences(messages, spec.placeholder_token)
        payload = modalities_payload.get(modality_id)

        if occurrences == 0 and payload is None:
            continue

        if occurrences > 0 and payload is None:
            raise ValueError(
                f"Prompt references placeholder `{spec.placeholder_token}` for modality `{modality_id}` "
                f"but no payload was provided."
            )

        if occurrences == 0 and payload is not None:
            logger.warning(
                "Received payload for modality `{}` but found no `{}` placeholder in prompt.",
                modality_id,
                spec.placeholder_token,
            )

        reserved_tokens = _resolve_reserved_tokens(spec, occurrences, payload)
        plans[modality_id] = ModalityPlaceholderPlan(
            modality_id=modality_id,
            placeholder_token=spec.placeholder_token,
            occurrences=occurrences,
            reserved_tokens=reserved_tokens,
            payload=payload,
        )

    # Warn about placeholders with no registered modality
    if placeholder_lookup:
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            start = 0
            while True:
                idx = content.find("<", start)
                if idx == -1:
                    break
                end = content.find(">", idx + 1)
                if end == -1:
                    break
                token = content[idx : end + 1]
                if token.endswith("_pad>") and token not in placeholder_lookup:
                    logger.warning(
                        "Encountered placeholder `{}` with no configured modality. "
                        "It will remain unmodified.",
                        token,
                    )
                start = end + 1

    return plans


def plan_modalities_for_batch(
    prompts: Sequence[Sequence[Mapping[str, Any]]],
    modalities_payloads: Sequence[Mapping[str, Any]],
    specs: Mapping[str, ModalitySpec],
) -> List[Dict[str, ModalityPlaceholderPlan]]:
    plans: List[Dict[str, ModalityPlaceholderPlan]] = []
    for messages, payload in zip(prompts, modalities_payloads):
        plans.append(plan_modalities_for_prompt(messages, payload, specs))
    return plans


def collect_embedding_spans(
    token_ids: Sequence[int],
    placeholder_token_id: int,
    reserved_tokens: Sequence[int],
) -> List[Tuple[int, int]]:
    positions = [idx for idx, token in enumerate(token_ids) if token == placeholder_token_id]
    if not reserved_tokens:
        return []
    if len(positions) < sum(reserved_tokens):
        raise ValueError(
            "Not enough placeholder tokens found during tokenization. "
            "Required {}, found {}.".format(sum(reserved_tokens), len(positions))
        )
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for length in reserved_tokens:
        if length <= 0:
            spans.append((-1, 0))
            continue
        start_pos_index = cursor
        end_pos_index = cursor + length - 1
        if end_pos_index >= len(positions):
            raise ValueError(
                "Placeholder token span exceeds available tokens. Requested length {} starting at index {}.".format(
                    length, cursor
                )
            )
        start = positions[start_pos_index]
        end = positions[end_pos_index]
        spans.append((start, end - start + 1))
        cursor += length
    if cursor < len(positions):
        logger.warning(
            "Unused placeholder tokens detected during span collection: %d extra tokens ignored.",
            len(positions) - cursor,
        )
    return spans


__all__ = [
    "ModalityHandlerSpec",
    "ModalityTrainableSpec",
    "ModalitySpec",
    "ModalityPlaceholderPlan",
    "ModalityEmbeddingSpan",
    "normalize_modalities_config",
    "build_placeholder_lookup",
    "plan_modalities_for_prompt",
    "plan_modalities_for_batch",
    "collect_embedding_spans",
]
