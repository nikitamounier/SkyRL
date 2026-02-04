# This code is adapted from OpenRLHF and OpenReasonerZero
# https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/models.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from typing import List, Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import numpy as np
from skyrl_train.distributed.ulysses.utils import (
    ulysses_pad_and_slice_inputs,
    gather_outputs_and_unpad,
    slice_input_tensor,
)
from skyrl_train.utils.torch_utils import chunked_entropy_from_logits, logprobs_from_logits
from flash_attn.bert_padding import pad_input, unpad_input
from packaging.version import Version
from skyrl_train.dataset.modalities import normalize_modalities_config
from skyrl_train.modalities import ModalitiesManager
from skyrl_train.modalities.types import SampleModalityData
from skyrl_train.modalities.batching import build_modality_batches


class HFModelWrapper(nn.Module):
    """
    Base class for wrapped HF models in reinforcement learning.

    This class serves as a foundation for implementing various model roles.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        exclude_modules (list, optional): List of modules to exclude from applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        # TODO(shu): combine all LoRA specific configs into one place?
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        exclude_modules=None,
        ds_config=None,
        device_map=None,
        temperature=1.0,
        use_liger_kernel=False,
        sequence_parallel_size=1,
        use_sample_packing: bool = False,
        use_torch_compile: bool = False,
        modalities_config: Optional[dict] = None,
        freeze_base_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.sequence_parallel_size = sequence_parallel_size
        self.attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
        self.use_sample_packing = use_sample_packing
        # packing samples using Flash Attention 2
        if use_sample_packing:
            assert (
                self.attn_implementation == "flash_attention_2"
            ), "Flash attention 2 should be used for `use_sample_packing`"

        self.modality_specs = normalize_modalities_config(modalities_config)
        self.modalities_manager: Optional[ModalitiesManager] = (
            ModalitiesManager(self.modality_specs) if self.modality_specs else None
        )
        self._modality_encoder_modules = nn.ModuleDict()
        self._modality_projector_modules = nn.ModuleDict()

        if isinstance(pretrain_or_model, str):
            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                if bf16 and ds_config["torch_autocast"]["enabled"]:
                    # The model’s dtype on initialization follows the config passed to `HfDeepSpeedConfig`,
                    # regardless of the `torch_dtype` specified in `from_pretrained`.
                    # To align with this behavior, we temporarily set `bf16` to True in a copied config.
                    # Note: this does NOT affect the config passed to `deepspeed.initialize()`.
                    ds_config = deepcopy(ds_config)
                    ds_config["bf16"] = {"enabled": True}
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None  # noqa: F841

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map=device_map,
            )

            # gpt oss
            if Version(transformers.__version__) >= Version("4.56.2"):
                from transformers import GptOssConfig

                if isinstance(self.model.config, GptOssConfig):
                    # patch attention with Unsloth's flex attn
                    from skyrl_train.patches.gptoss.patch_transformers import (
                        custom_attention,
                        custom_attention_mask,
                        patch_GptOssAttention,
                    )
                    from transformers import AttentionInterface, AttentionMaskInterface

                    AttentionInterface.register("custom_flex", custom_attention)
                    AttentionMaskInterface.register("custom_flex", custom_attention_mask)
                    # set attention implementation to be `custom_flex`
                    self.model.set_attn_implementation("custom_flex")
                    self.attn_implementation = "custom_flex"
                    # NOTE: Even though we set a custom attn implementation, we
                    # also patch the full attention function for GPT OSS
                    patch_GptOssAttention()

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    exclude_modules=exclude_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logger.info("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

        if self.modalities_manager is not None:
            # Ensure the underlying HF module owns modality components so optimizer/state dicts capture them.
            if hasattr(self.model, "_skyrl_modality_encoders"):
                self._base_modality_encoder_modules = getattr(self.model, "_skyrl_modality_encoders")
            else:
                self._base_modality_encoder_modules = nn.ModuleDict()
                self.model.add_module("_skyrl_modality_encoders", self._base_modality_encoder_modules)

            if hasattr(self.model, "_skyrl_modality_projections"):
                self._base_modality_projection_modules = getattr(self.model, "_skyrl_modality_projections")
            else:
                self._base_modality_projection_modules = nn.ModuleDict()
                self.model.add_module("_skyrl_modality_projections", self._base_modality_projection_modules)
        else:
            self._base_modality_encoder_modules = None
            self._base_modality_projection_modules = None

        if self.modalities_manager is not None:
            for modality_id, role, module in self.modalities_manager.iter_handler_modules():
                if not isinstance(module, nn.Module):
                    continue
                if role == "encoder":
                    self._modality_encoder_modules[modality_id] = module
                    if self._base_modality_encoder_modules is not None:
                        self._base_modality_encoder_modules[modality_id] = module
                elif role == "projection":
                    self._modality_projector_modules[modality_id] = module
                    if self._base_modality_projection_modules is not None:
                        self._base_modality_projection_modules[modality_id] = module

        if freeze_base_model:
            self._freeze_base_parameters()

        self._freeze_base_model = freeze_base_model

        # TODO (sumanthrh): do the same for `logprobs_from_logits` and test.
        # Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/#efficient-solution
        self.chunked_entropy_from_logits_fn = (
            torch.compile(chunked_entropy_from_logits, dynamic=True)
            if use_torch_compile
            else chunked_entropy_from_logits
        )

    def _freeze_base_parameters(self) -> None:
        """Freeze non-modality parameters on the underlying HF model."""
        modality_prefixes = ("_skyrl_modality_encoders.", "_skyrl_modality_projections.")

        for name, param in self.model.named_parameters():
            if name.startswith(modality_prefixes):
                continue
            param.requires_grad = False

        # Re-apply modality trainability based on specs
        if self.modalities_manager is not None:
            for modality_id, role, module in self.modalities_manager.iter_handler_modules():
                if not isinstance(module, nn.Module):
                    continue
                spec = self.modalities_manager._specs.get(modality_id)
                if spec is None:
                    continue
                trainable = spec.trainable.encoder if role == "encoder" else spec.trainable.projection
                for param in module.parameters():
                    param.requires_grad = trainable

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        modalities_metadata: Optional[List[SampleModalityData]] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        modalities_metadata = kwargs.pop("modalities_metadata", modalities_metadata)
        provided_inputs_embeds = kwargs.pop("inputs_embeds", None)
        generate_args = {
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "min_p": kwargs.get("min_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        use_modalities = self.modalities_manager is not None and modalities_metadata is not None
        if provided_inputs_embeds is not None:
            generate_args["inputs_embeds"] = provided_inputs_embeds
        elif use_modalities:
            inputs_embeds, _ = self.prepare_inputs_embeds(
                input_ids, modalities_metadata=modalities_metadata, update_metadata=False
            )
            generate_args["inputs_embeds"] = inputs_embeds
        else:
            generate_args["input_ids"] = input_ids

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """
        Process generated sequences to create attention masks and action masks.

        Args:
            sequences (torch.Tensor): Generated sequence tensor
            input_len (int): Length of the input sequence
            eos_token_id (int): Token ID for the end-of-sequence token
            pad_token_id (int): Token ID for the padding token

        Returns:
            tuple: A tuple containing three elements:
                - sequences: Original sequence
                - attention_mask: Attention mask indicating valid token positions
                - action_mask: Action mask indicating valid action token positions
        """
        # Create initial attention mask by marking positions that are neither EOS nor padding tokens
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find the position of the last valid token in each sequence
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

        # Handle cases where EOS tokens might appear in the middle of the prompt (for Llama3 and Qwen2 models)
        # Find the position of the first valid token in each sequence
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # Create position mask
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # Generate final attention mask, keeping only positions between first and last valid tokens
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # In reinforcement learning, the state transition is represented as:
        # state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # Generate state sequence from input_len-1 to second-to-last token
        state_seq = sequences[:, input_len - 1 : -1]
        # Generate action mask indicating valid action token positions
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def _safe_get_embeddings(self, embedding_layer, input_ids):
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # FSDP2 Check: DTensor
        try:
            try:
                from torch.distributed.tensor import DTensor
            except ImportError:
                from torch.distributed._tensor import DTensor

            if hasattr(embedding_layer, "weight") and isinstance(embedding_layer.weight, DTensor):
                full_weight = embedding_layer.weight.full_tensor()
                return F.embedding(
                    input_ids,
                    full_weight,
                    padding_idx=embedding_layer.padding_idx,
                    max_norm=embedding_layer.max_norm,
                    norm_type=embedding_layer.norm_type,
                    scale_grad_by_freq=embedding_layer.scale_grad_by_freq,
                    sparse=embedding_layer.sparse,
                )
        except ImportError:
            pass

        # FSDP1 Check: Storage size 0 (Sharded)
        if hasattr(embedding_layer, "weight") and embedding_layer.weight.storage().size() == 0:
            # Try gathering params.
            # We use self.model (the HF model) as the root for summoning.
            # This will summon all params in self.model.
            with FSDP.summon_full_params(self.model, writeback=False):
                return embedding_layer(input_ids)

        return embedding_layer(input_ids)

    def _mask_modality_tokens(
        self,
        input_ids: torch.LongTensor,
        modalities_metadata: Optional[List[SampleModalityData]],
    ) -> torch.LongTensor:
        if modalities_metadata is None:
            return input_ids
        if not any(meta.embedding_spans for meta in modalities_metadata):
            return input_ids

        safe_ids = input_ids.clone()
        pad_token_id = getattr(self.model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.model.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        for sample_idx, meta in enumerate(modalities_metadata):
            for spans in meta.embedding_spans.values():
                for span in spans:
                    start = span.token_start
                    length = span.token_length
                    if start < 0 or length <= 0:
                        continue
                    end = start + length
                    if end > safe_ids.size(1):
                        raise ValueError(
                            f"Modality span ({start}, {length}) exceeds sequence length {safe_ids.size(1)}."
                        )
                    safe_ids[sample_idx, start:end] = pad_token_id
        return safe_ids

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        modalities_metadata: Optional[List[SampleModalityData]] = None,
        *,
        update_metadata: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[SampleModalityData]]]:
        """
        Build input embeddings, optionally injecting modality-specific projections in place of placeholder tokens.

        Args:
            input_ids: Token ids for the batch (batch, seq_len)
            modalities_metadata: Per-sample modality metadata describing placeholder spans and payloads.
            update_metadata: Whether to allow modality handlers to update the supplied metadata with encoder/projector outputs.

        Returns:
            Tuple of (inputs_embeds, modalities_metadata). The metadata list is returned unchanged unless
            `update_metadata=True`, in which case the same list (potentially mutated) is returned.
        """
        embedding_layer = self.model.get_input_embeddings()
        if embedding_layer is None:
            raise RuntimeError("Underlying model does not expose input embeddings.")
        safe_input_ids = self._mask_modality_tokens(input_ids, modalities_metadata)
        base_embeddings = self._safe_get_embeddings(embedding_layer, safe_input_ids)

        if self.modalities_manager is None or not modalities_metadata:
            return base_embeddings, modalities_metadata

        batch_size = input_ids.size(0)
        if len(modalities_metadata) != batch_size:
            raise ValueError(
                f"Expected {batch_size} modality metadata entries, but received {len(modalities_metadata)}."
            )

        samples_metadata: List[SampleModalityData] = []
        for idx, meta in enumerate(modalities_metadata):
            if not isinstance(meta, SampleModalityData):
                raise TypeError(
                    f"Modalities metadata at index {idx} must be a SampleModalityData instance; got {type(meta).__name__}."
                )
            samples_metadata.append(meta)

        if not any(meta.plans for meta in samples_metadata):
            return base_embeddings, modalities_metadata

        modality_batches = build_modality_batches(samples_metadata)
        if not modality_batches:
            return base_embeddings, modalities_metadata

        replacements = self.modalities_manager.compute_embeddings(
            modality_batches,
            samples_metadata,
            target_device=base_embeddings.device,
            target_dtype=base_embeddings.dtype,
            non_blocking=True,
            update_metadata=update_metadata,
        )
        if not replacements:
            return base_embeddings, modalities_metadata

        updated_embeddings = base_embeddings.clone()
        seq_len = updated_embeddings.size(1)
        hidden_size = updated_embeddings.size(2)
        for sample_idx, span_list in replacements.items():
            if sample_idx >= batch_size:
                raise ValueError(
                    f"Replacement requested for sample index {sample_idx}, but batch size is {batch_size}."
                )
            for (start, length), tensor in span_list:
                if start < 0 or length <= 0:
                    raise ValueError(
                        f"Invalid replacement span (start={start}, length={length}) for sample {sample_idx}."
                    )
                if start + length > seq_len:
                    raise ValueError(
                        f"Replacement span (start={start}, length={length}) exceeds sequence length {seq_len}."
                    )
                if tensor.shape[0] != length:
                    raise ValueError(
                        f"Projected embedding length mismatch: expected {length}, got {tensor.shape[0]}."
                    )
                if tensor.shape[1] != hidden_size:
                    raise ValueError(
                        f"Projected embedding hidden size mismatch: expected {hidden_size}, got {tensor.shape[1]}."
                    )
                updated_embeddings[sample_idx, start : start + length, :] = tensor

        return updated_embeddings, samples_metadata if update_metadata else modalities_metadata

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_output=False,
        compute_entropy=False,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        modalities_metadata: Optional[List[SampleModalityData]] = None,
        update_modalities_metadata: bool = False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if attention_mask is None:
            attention_mask = torch.ones_like(sequences, dtype=torch.long)

        modalities_metadata = self._shift_modalities_metadata_for_padding(
            sequences=sequences,
            attention_mask=attention_mask,
            num_actions=num_actions,
            modalities_metadata=modalities_metadata,
        )

        sequences_safe = self._mask_modality_tokens(sequences, modalities_metadata)

        metadata_result = modalities_metadata
        if inputs_embeds is None:
            inputs_embeds, metadata_result = self.prepare_inputs_embeds(
                sequences_safe,
                modalities_metadata=modalities_metadata,
                update_metadata=update_modalities_metadata,
            )

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        sequences_fwd = sequences_safe
        position_ids_fwd = position_ids
        attention_mask_fwd = attention_mask
        inputs_embeds_fwd = inputs_embeds
        if self.use_sample_packing:
            with torch.no_grad():
                # Removes padding to get a packed tensor. `unpad_input` expects 3 dimensional tensor so we unsqueeze first
                sequences_fwd, nnz_indices, _, _, _ = unpad_input(
                    sequences.unsqueeze(-1), attention_mask=attention_mask
                )
                # (nnz, 1) -> (1, nnz)
                sequences_fwd = sequences_fwd.transpose(0, 1)
                position_ids_fwd, _, _, _, _ = unpad_input(position_ids.unsqueeze(-1), attention_mask)
                # (nnz, 1) -> (1, nnz)
                position_ids_fwd = position_ids_fwd.transpose(0, 1)
                attention_mask_fwd = None  # no attention mask with FA 2

            if inputs_embeds_fwd is not None:
                inputs_embeds_unpacked, _, _, _, _ = unpad_input(inputs_embeds_fwd, attention_mask=attention_mask)
                inputs_embeds_fwd = inputs_embeds_unpacked.unsqueeze(0)

        sequences_rolled = torch.roll(sequences_fwd, shifts=-1, dims=1)
        if self.sequence_parallel_size > 1:
            # NOTE: don't pass any attn mask with sample packing
            attention_mask_fwd = None if self.use_sample_packing else attention_mask_fwd

            # slice for sequence parallelism
            # (bsz, seqlen) -> (bsz, seqlen//sp_size)
            sequences_fwd, position_ids_fwd, attention_mask_fwd, pad_size = ulysses_pad_and_slice_inputs(
                sequences_fwd, position_ids_fwd, attention_mask_fwd, self.sequence_parallel_size
            )
            sequences_rolled, _, _, _ = ulysses_pad_and_slice_inputs(
                sequences_rolled, None, None, self.sequence_parallel_size
            )
            if inputs_embeds_fwd is not None:
                if pad_size > 0:
                    inputs_embeds_fwd = F.pad(inputs_embeds_fwd, (0, 0, 0, pad_size))
                inputs_embeds_fwd = slice_input_tensor(inputs_embeds_fwd, dim=1, padding=False)

        # NOTE (sumanthrh): Once we have position_ids, we don't need attention mask with flash attention.
        if self.use_sample_packing and self.attn_implementation == "flash_attention_2":
            # NOTE (sumanthrh): Don't use attention mask. position_ids is enough.
            # Not using attention mask leads to higher perf since flash attention varlen func is enabled
            if inputs_embeds_fwd is not None:
                output = self.model(attention_mask=None, position_ids=position_ids_fwd, inputs_embeds=inputs_embeds_fwd)
            else:
                output = self.model(sequences_fwd, attention_mask=None, position_ids=position_ids_fwd)
        else:
            if inputs_embeds_fwd is not None:
                output = self.model(
                    attention_mask=attention_mask_fwd,
                    position_ids=position_ids_fwd,
                    inputs_embeds=inputs_embeds_fwd,
                )
            else:
                output = self.model(sequences_fwd, attention_mask=attention_mask_fwd, position_ids=position_ids_fwd)

        logits_BSV = output["logits"]
        logits_BSV.div_(temperature)

        # NOTE: this is slightly inaccurate with sample packing because last token from nth seq -> first token of n+1th seq loss is added.
        log_probs = logprobs_from_logits(
            logits_BSV,
            sequences_rolled,
            inplace_backward=True,
        )

        # gather output if sp > 1
        if self.sequence_parallel_size > 1:
            dim = log_probs.ndim - 1
            log_probs = gather_outputs_and_unpad(
                log_probs, gather_dim=dim, unpad_dim=dim, padding_size=pad_size
            )  # shape can be (1, nnz) - with packing or (B, S) - without packing

        if self.use_sample_packing:
            # add padding back - postprocess logprobs to be compatible with original tensor
            batch_size, seqlen = attention_mask.shape
            # (1, nnz-1) -> (batch_size, seqlen). Pad token ID used by flash attention is 0.
            log_probs = pad_input(
                log_probs.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
            ).squeeze(-1)

        if compute_entropy:
            # entropy calculation as a metric - we use no grad
            # For sample packing: entropy is calculated on unpacked data, so no attention mask needed
            # For non-sample packing: pass the attention mask to exclude padding tokens
            entropy_mask = None
            if not self.use_sample_packing:
                # Non-sample packing: pass attention mask to handle padding
                # Use attention_mask_fwd which may be sliced (if sequence_parallel_size > 1) or full
                entropy_mask = attention_mask_fwd

            entropy_BS = self.chunked_entropy_from_logits_fn(
                logits_BSV, requires_grad=False, attention_mask=entropy_mask
            )

            if self.sequence_parallel_size > 1:
                dim = entropy_BS.ndim - 1
                entropy_BS = gather_outputs_and_unpad(
                    entropy_BS, gather_dim=dim, unpad_dim=dim, padding_size=pad_size
                )  # shape can be (1, nnz) - with packing or (B,S) - without packing
            if self.use_sample_packing:
                entropy_BS = pad_input(
                    entropy_BS.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
                ).squeeze(
                    -1
                )  # (1, nnz) -> (B, S)

            output["entropy"] = entropy_BS

        if metadata_result is not None:
            try:
                output["modalities_metadata"] = metadata_result
            except TypeError:
                setattr(output, "modalities_metadata", metadata_result)

        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                num_actions = np.array(num_actions)
        action_log_probs = log_probs[:, -num_actions - 1 : -1]

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def _shift_modalities_metadata_for_padding(
        self,
        *,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        num_actions: Union[int, list[int]],
        modalities_metadata: Optional[List[SampleModalityData]],
    ) -> Optional[List[SampleModalityData]]:
        """Shift modality embedding spans to account for left-padding in training batches."""
        if modalities_metadata is None:
            return None

        if not isinstance(num_actions, int):
            return modalities_metadata

        if num_actions <= 0:
            return modalities_metadata

        seq_len = sequences.size(1)
        max_prompt_len = seq_len - num_actions
        if max_prompt_len <= 0:
            return modalities_metadata

        response_mask = attention_mask[:, -num_actions:]
        response_lengths = response_mask.sum(dim=1)
        prompt_lengths = attention_mask.sum(dim=1) - response_lengths
        pad_lengths = max_prompt_len - prompt_lengths

        if torch.all(pad_lengths == 0):
            return modalities_metadata

        shifted_metadata: List[SampleModalityData] = []
        for idx, meta in enumerate(modalities_metadata):
            pad_len = int(pad_lengths[idx].item())
            if pad_len <= 0 or not meta.embedding_spans:
                shifted_metadata.append(meta)
                continue

            if hasattr(meta, "clone"):
                meta = meta.clone()
            else:
                meta = deepcopy(meta)

            for spans in meta.embedding_spans.values():
                for span in spans:
                    span.token_start += pad_len

            shifted_metadata.append(meta)

        return shifted_metadata
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def _get_critic_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="value_head",
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
    modalities_config: Optional[dict] = None,
):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.sequence_parallel_size = sequence_parallel_size
            self.use_sample_packing = use_sample_packing
            if use_sample_packing:
                assert (
                    config._attn_implementation == "flash_attention_2"
                ), "Flash attention must be used with sample packing"

            if self.sequence_parallel_size > 1:
                logger.info("Critic model using sequence parallelism with size: ", self.sequence_parallel_size)

            self.modality_specs = normalize_modalities_config(modalities_config)
            self.modalities_manager: Optional[ModalitiesManager] = (
                ModalitiesManager(self.modality_specs) if self.modality_specs else None
            )
            self._modality_encoder_modules = nn.ModuleDict()
            self._modality_projector_modules = nn.ModuleDict()
            if self.modalities_manager is not None:
                if hasattr(self, "_skyrl_modality_encoders"):
                    base_encoders = getattr(self, "_skyrl_modality_encoders")
                else:
                    base_encoders = nn.ModuleDict()
                    self.add_module("_skyrl_modality_encoders", base_encoders)
                if hasattr(self, "_skyrl_modality_projections"):
                    base_projections = getattr(self, "_skyrl_modality_projections")
                else:
                    base_projections = nn.ModuleDict()
                    self.add_module("_skyrl_modality_projections", base_projections)
            else:
                base_encoders = None
                base_projections = None
            if self.modalities_manager is not None:
                for modality_id, role, module in self.modalities_manager.iter_handler_modules():
                    if not isinstance(module, nn.Module):
                        continue
                    if role == "encoder":
                        self._modality_encoder_modules[modality_id] = module
                        if base_encoders is not None:
                            base_encoders[modality_id] = module
                    elif role == "projection":
                        self._modality_projector_modules[modality_id] = module
                        if base_projections is not None:
                            base_projections[modality_id] = module

        def _safe_get_embeddings(self, embedding_layer, input_ids):
            import torch.distributed as dist
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # FSDP2 Check: DTensor
            try:
                try:
                    from torch.distributed.tensor import DTensor
                except ImportError:
                    from torch.distributed._tensor import DTensor

                if hasattr(embedding_layer, "weight") and isinstance(embedding_layer.weight, DTensor):
                    full_weight = embedding_layer.weight.full_tensor()
                    return F.embedding(
                        input_ids,
                        full_weight,
                        padding_idx=embedding_layer.padding_idx,
                        max_norm=embedding_layer.max_norm,
                        norm_type=embedding_layer.norm_type,
                        scale_grad_by_freq=embedding_layer.scale_grad_by_freq,
                        sparse=embedding_layer.sparse,
                    )
            except ImportError:
                pass

            # FSDP1 Check: Storage size 0 (Sharded)
            if hasattr(embedding_layer, "weight") and embedding_layer.weight.storage().size() == 0:
                # Try gathering params.
                # We use self (the model) as the root for summoning if it's wrapped,
                # but here self is the inner model.
                # However, if we call summon_full_params on self, it will summon sub-FSDP modules.
                # This handles the case where embeddings are in an auto-wrapped sub-FSDP unit.
                with FSDP.summon_full_params(self, writeback=False):
                    return embedding_layer(input_ids)

            return embedding_layer(input_ids)

        def _mask_modality_tokens(
            self,
            input_ids: torch.LongTensor,
            modalities_metadata: Optional[List[SampleModalityData]],
        ) -> torch.LongTensor:
            if modalities_metadata is None:
                return input_ids
            if not any(meta.embedding_spans for meta in modalities_metadata):
                return input_ids

            safe_ids = input_ids.clone()
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.config, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = 0

            for sample_idx, meta in enumerate(modalities_metadata):
                for spans in meta.embedding_spans.values():
                    for span in spans:
                        start = span.token_start
                        length = span.token_length
                        if start < 0 or length <= 0:
                            continue
                        end = start + length
                        if end > safe_ids.size(1):
                            raise ValueError(
                                f"Modality span ({start}, {length}) exceeds sequence length {safe_ids.size(1)}."
                            )
                        safe_ids[sample_idx, start:end] = pad_token_id
            return safe_ids

        def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            modalities_metadata: Optional[List[SampleModalityData]] = None,
            *,
            update_metadata: bool = False,
        ) -> Tuple[torch.Tensor, Optional[List[SampleModalityData]]]:
            embedding_layer = self.get_input_embeddings()
            if embedding_layer is None:
                raise RuntimeError("Underlying model does not expose input embeddings.")
            safe_input_ids = self._mask_modality_tokens(input_ids, modalities_metadata)
            base_embeddings = self._safe_get_embeddings(embedding_layer, safe_input_ids)

            if self.modalities_manager is None or not modalities_metadata:
                return base_embeddings, modalities_metadata

            batch_size = input_ids.size(0)
            if len(modalities_metadata) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} modality metadata entries, got {len(modalities_metadata)}."
                )

            samples_metadata: List[SampleModalityData] = []
            for idx, meta in enumerate(modalities_metadata):
                if not isinstance(meta, SampleModalityData):
                    raise TypeError(
                        f"Modalities metadata at index {idx} must be a SampleModalityData instance; "
                        f"got {type(meta).__name__}."
                    )
                samples_metadata.append(meta)

            if not any(meta.plans for meta in samples_metadata):
                return base_embeddings, modalities_metadata

            modality_batches = build_modality_batches(samples_metadata)
            if not modality_batches:
                return base_embeddings, modalities_metadata

            replacements = self.modalities_manager.compute_embeddings(
                modality_batches,
                samples_metadata,
                target_device=base_embeddings.device,
                target_dtype=base_embeddings.dtype,
                non_blocking=True,
                update_metadata=update_metadata,
            )
            if not replacements:
                return base_embeddings, modalities_metadata

            updated_embeddings = base_embeddings.clone()
            seq_len = updated_embeddings.size(1)
            hidden_size = updated_embeddings.size(2)
            for sample_idx, span_list in replacements.items():
                if sample_idx >= batch_size:
                    raise ValueError(
                        f"Replacement requested for sample index {sample_idx}, but batch size is {batch_size}."
                    )
                for (start, length), tensor in span_list:
                    if start < 0 or length <= 0:
                        raise ValueError(
                            f"Invalid replacement span (start={start}, length={length}) for sample {sample_idx}."
                        )
                    if start + length > seq_len:
                        raise ValueError(
                            f"Replacement span (start={start}, length={length}) exceeds sequence length {seq_len}."
                        )
                    if tensor.shape[0] != length:
                        raise ValueError(
                            f"Projected embedding length mismatch: expected {length}, got {tensor.shape[0]}."
                        )
                    if tensor.shape[1] != hidden_size:
                        raise ValueError(
                            f"Projected embedding hidden size mismatch: expected {hidden_size}, got {tensor.shape[1]}."
                        )
                    updated_embeddings[sample_idx, start : start + length, :] = tensor

            return updated_embeddings, samples_metadata if update_metadata else modalities_metadata

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            *,
            inputs_embeds: Optional[torch.Tensor] = None,
            modalities_metadata: Optional[List[SampleModalityData]] = None,
            update_modalities_metadata: bool = False,
        ) -> torch.Tensor:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            input_ids_safe = self._mask_modality_tokens(input_ids, modalities_metadata)

            metadata_result = modalities_metadata
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds, metadata_result = self.prepare_inputs_embeds(
                    input_ids_safe,
                    modalities_metadata=modalities_metadata,
                    update_metadata=update_modalities_metadata,
                )

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            input_ids_fwd = input_ids_safe
            position_ids_fwd = position_ids
            attention_mask_fwd = attention_mask
            inputs_embeds_fwd = inputs_embeds

            if self.use_sample_packing:
                with torch.no_grad():
                    # remove padding. `unpad_input` expects 3 dimensional tensor
                    input_ids_fwd, nnz_indices, _, _, _ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    input_ids_fwd = input_ids_fwd.transpose(0, 1)
                    position_ids_fwd, _, _, _, _ = unpad_input(
                        position_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    position_ids_fwd = position_ids_fwd.transpose(0, 1)
                    # don't use attention mask with FA2
                    attention_mask_fwd = None
                if inputs_embeds_fwd is not None:
                    inputs_embeds_unpacked, _, _, _, _ = unpad_input(inputs_embeds_fwd, attention_mask=attention_mask)
                    inputs_embeds_fwd = inputs_embeds_unpacked.unsqueeze(0)

            if self.sequence_parallel_size > 1:
                assert self.use_sample_packing, "sample packing must be true for sequence parallelism"
                # don't pass any attention mask for flash attention 2. this will save an all gather.
                attention_mask_fwd = None if self.config._attn_implementation == "flash_attention_2" else attention_mask
                # slice for sequence parallelism
                # (bsz, seqlen) -> (bsz, seqlen//sp_size)
                input_ids_fwd, position_ids_fwd, attention_mask_fwd, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_fwd, position_ids_fwd, attention_mask_fwd, self.sequence_parallel_size
                )
                if inputs_embeds_fwd is not None:
                    if pad_size > 0:
                        inputs_embeds_fwd = F.pad(inputs_embeds_fwd, (0, 0, 0, pad_size))
                    inputs_embeds_fwd = slice_input_tensor(inputs_embeds_fwd, dim=1, padding=False)

            if self.sequence_parallel_size > 1 and self.config._attn_implementation == "flash_attention_2":
                if inputs_embeds_fwd is not None:
                    outputs = getattr(self, self.base_model_prefix)(
                        position_ids=position_ids_fwd, inputs_embeds=inputs_embeds_fwd
                    )
                else:
                    outputs = getattr(self, self.base_model_prefix)(input_ids_fwd, position_ids=position_ids_fwd)
            else:
                if inputs_embeds_fwd is not None:
                    outputs = getattr(self, self.base_model_prefix)(
                        attention_mask=attention_mask_fwd, position_ids=position_ids_fwd, inputs_embeds=inputs_embeds_fwd
                    )
                else:
                    outputs = getattr(self, self.base_model_prefix)(
                        input_ids_fwd, attention_mask=attention_mask_fwd, position_ids=position_ids_fwd
                    )
            last_hidden_states_BSH = outputs["last_hidden_state"]

            if self.sequence_parallel_size > 1:
                last_hidden_states_SH = last_hidden_states_BSH.squeeze(0)
                # (seqlen*bsz//sp_size, 1) -> (seqlen*bsz, 1)
                last_hidden_states_SH = gather_outputs_and_unpad(
                    last_hidden_states_SH, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
                last_hidden_states_BSH = last_hidden_states_SH.unsqueeze(0)

            values_BSH = getattr(self, self.value_head_prefix)(last_hidden_states_BSH)

            if self.use_sample_packing:
                # add padding back - postprocess logits to be compatible with original tensors
                batch_size, seqlen = attention_mask.shape
                # (1, nnz, 1) -> (nnz, 1) -> (batch_size, seqlen, 1)
                values_BSH = pad_input(values_BSH.squeeze(0), indices=nnz_indices, batch=batch_size, seqlen=seqlen)

            values = values_BSH.squeeze(-1)[:, :-1]

            if num_actions is None:
                assert return_output
                return outputs

            action_values = values[:, -num_actions:]

            if metadata_result is not None:
                try:
                    outputs["modalities_metadata"] = metadata_result
                except TypeError:
                    setattr(outputs, "modalities_metadata", metadata_result)

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    exclude_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
    modalities_config: Optional[dict] = None,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Type of sequence classification model. Only `critic` is supported.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert model_type == "critic", f"Only model_type critic is supported, got: {model_type}."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_critic_model(
        base_pretrained_class,
        base_class,
        value_head_prefix,
        sequence_parallel_size=sequence_parallel_size,
        use_sample_packing=use_sample_packing,
        modalities_config=modalities_config,
    )

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        logger.info("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            import deepspeed

            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model
