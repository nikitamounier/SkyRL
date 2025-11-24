from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.modalities import (
    ModalityEmbeddingSpan,
    ModalityPlaceholderPlan,
    ModalitySpec,
    collect_embedding_spans,
)


@dataclass
class ProcessedPrompt:
    token_ids: List[int]
    attention_mask: List[int]
    embedding_spans: Dict[str, List[ModalityEmbeddingSpan]]
    expanded_messages: List[Dict[str, Any]]


class MultimodalPromptProcessor:
    """
    Expands modality placeholders in prompts and records the embedding spans for later replacement.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        modality_specs: Mapping[str, ModalitySpec],
        chat_template_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.modality_specs: Dict[str, ModalitySpec] = dict(modality_specs)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._placeholder_token_ids: Dict[str, int] = {}

        for modality_id, spec in self.modality_specs.items():
            token_id = self.tokenizer.convert_tokens_to_ids(spec.placeholder_token)
            if token_id == -1:
                raise ValueError(
                    f"Tokenizer does not know placeholder token `{spec.placeholder_token}` "
                    f"for modality `{modality_id}`. Add the token before using modalities."
                )
            self._placeholder_token_ids[modality_id] = token_id

    def process_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        plan: Mapping[str, ModalityPlaceholderPlan],
        *,
        add_generation_prompt: bool,
        chat_template: Optional[str] = None,
    ) -> ProcessedPrompt:
        expanded_messages = self._expand_messages_with_modalities(messages, plan)
        tokenization_kwargs = dict(self.chat_template_kwargs)
        tokenization_kwargs["add_generation_prompt"] = add_generation_prompt
        if chat_template is not None:
            tokenization_kwargs["chat_template"] = chat_template
        token_ids = self.tokenizer.apply_chat_template(
            expanded_messages,
            tokenize=True,
            **tokenization_kwargs,
        )
        attention_mask = [1] * len(token_ids)

        embedding_spans: Dict[str, List[ModalityEmbeddingSpan]] = {}
        for modality_id, placeholder_plan in plan.items():
            placeholder_token_id = self._placeholder_token_ids.get(modality_id)
            if placeholder_token_id is None:
                raise ValueError(f"Missing placeholder token id for modality `{modality_id}`.")
            spans = collect_embedding_spans(token_ids, placeholder_token_id, placeholder_plan.reserved_tokens)
            if not spans and placeholder_plan.occurrences > 0:
                raise ValueError(
                    f"Failed to locate placeholder tokens for modality `{modality_id}` after tokenization."
                )
            expected = sum(placeholder_plan.reserved_tokens)
            actual = sum(length for _, length in spans)
            if expected and actual != expected:
                raise ValueError(
                    f"Tokenized placeholder span length mismatch for modality `{modality_id}`: "
                    f"expected {expected}, got {actual}. Reserved lengths: {placeholder_plan.reserved_tokens}, "
                    f"token spans: {spans}"
                )
            embedding_spans[modality_id] = [
                ModalityEmbeddingSpan(
                    modality_id=modality_id,
                    occurrence_index=index,
                    token_start=start,
                    token_length=length,
                )
                for index, (start, length) in enumerate(spans)
            ]

        return ProcessedPrompt(
            token_ids=list(token_ids),
            attention_mask=attention_mask,
            embedding_spans=embedding_spans,
            expanded_messages=[dict(message) for message in expanded_messages],
        )

    def _expand_messages_with_modalities(
        self,
        messages: Sequence[Mapping[str, Any]],
        plan: Mapping[str, ModalityPlaceholderPlan],
    ) -> List[Dict[str, Any]]:
        if not plan:
            return [dict(message) for message in messages]

        expanded_messages: List[Dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                expanded_messages.append(dict(message))
                continue

            new_content = content
            for modality_id, placeholder_plan in plan.items():
                token = placeholder_plan.placeholder_token
                if token not in new_content:
                    continue
                new_content = self._expand_placeholder(
                    modality_id, new_content, placeholder_plan.placeholder_token, placeholder_plan.reserved_tokens
                )
            updated_message = dict(message)
            updated_message["content"] = new_content
            expanded_messages.append(updated_message)

        return expanded_messages

    def _expand_placeholder(
        self,
        modality_id: str,
        content: str,
        placeholder_token: str,
        reserved_tokens: Sequence[int],
    ) -> str:
        occurrences = content.count(placeholder_token)
        if occurrences < len(reserved_tokens):
            logger.warning(
                "Expected at least %d occurrences of placeholder `%s` for modality `%s`, found %d. "
                "Extra reserved tokens will be ignored.",
                len(reserved_tokens),
                placeholder_token,
                modality_id,
                occurrences,
            )
        result = content
        for reserved_length in reserved_tokens:
            replacement = placeholder_token * reserved_length
            result = result.replace(placeholder_token, replacement, 1)
        return result


__all__ = ["ProcessedPrompt", "MultimodalPromptProcessor"]
