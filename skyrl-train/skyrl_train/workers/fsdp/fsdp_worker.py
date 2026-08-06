import asyncio
import os
from typing import Dict, List

import ray
import torch
import torch.distributed
from transformers import AutoConfig
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import io

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl_train.model_wrapper import HFModelWrapper, get_llm_for_sequence_regression
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.distributed.fsdp_utils import fsdp_version, get_init_weight_context_manager
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RefWorkerBase,
)


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.policy.fsdp_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            model_config=self.cfg.trainer.policy.model,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        self._is_lora = self.cfg.trainer.policy.model.lora.rank > 0

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():

            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.trainer.policy.model.lora.rank,
                lora_alpha=self.cfg.trainer.policy.model.lora.alpha,
                lora_dropout=self.cfg.trainer.policy.model.lora.dropout,
                target_modules=self.cfg.trainer.target_modules,
                exclude_modules=self.cfg.trainer.exclude_modules,
                sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
                modalities_config=self.cfg.trainer.modalities,
                freeze_base_model=self.cfg.trainer.policy.model.get("freeze_base_model", False),
            )
            # in-place patch
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.trainer.gradient_checkpointing:
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

        self.use_cuda_ipc = False
        if self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all:
            self.use_cuda_ipc = True

        # Log trainable parameter count (rank 0) so LoRA / memory-module runs can
        # be compared on a like-for-like trainable-param budget.
        try:
            if torch.distributed.get_rank() == 0:
                _trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                _total = sum(p.numel() for p in self.model.model.parameters())
                print(
                    f"[trainable-params] trainable={_trainable:,} ({_trainable/1e6:.1f}M) "
                    f"total={_total:,} ({_total/1e6:.1f}M) lora_rank={self.cfg.trainer.policy.model.lora.rank}",
                    flush=True,
                )
        except Exception as _e:
            print(f"[trainable-params] count failed: {_e}", flush=True)

    async def _save_lora_adapters_and_sync(self, peft_model, lora_sync_path, inference_engine_client):
        """Collect LoRA parameters, save and call inference engine to load."""
        import os
        import json
        from dataclasses import asdict
        from safetensors.torch import save_file
        from skyrl_train.distributed.fsdp_utils import collect_lora_params

        import shutil

        lora_params = collect_lora_params(module=self.model.model)

        if torch.distributed.get_rank() == 0:
            # Write each sync to a UNIQUE directory. vLLM can cache a parsed adapter by its disk
            # path, so reusing one path serves the first (step-0, empty) weights forever -> frozen
            # rollout. A fresh path every step forces vLLM to re-parse the current weights.
            self._lora_sync_counter = getattr(self, "_lora_sync_counter", 0) + 1
            step_dir = os.path.join(lora_sync_path, f"v{self._lora_sync_counter}")
            os.makedirs(step_dir, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(step_dir, "adapter_model.safetensors"))
            with io.open(os.path.join(step_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine. `lora_disk_load` is a specific identifier
            # to tell the inference engine to extract the `lora_disk_path`.
            lora_request = {
                "names": ["lora_disk_load"],
                "extras": [{"lora_disk_path": step_dir}],
            }
            await inference_engine_client.update_named_weights(lora_request)

            # clean up the previous step's directory to avoid unbounded disk growth
            prev_dir = os.path.join(lora_sync_path, f"v{self._lora_sync_counter - 1}")
            if os.path.isdir(prev_dir):
                shutil.rmtree(prev_dir, ignore_errors=True)

        torch.distributed.barrier()

    async def _merge_lora_into_base_and_broadcast(self, peft_model, inference_engine_client, generator_dtype):
        """Bake the LoRA delta into the base weights and push the merged weights into vLLM's BASE model.

        vLLM does NOT apply LoRA on the ``prompt_embeds`` generation path we use to inject the cell
        embedding, so the synced adapter never affected rollouts (frozen base-model rollout, identical
        rewards across LRs). Only LoRA-target weights change during LoRA training; the rest of the base
        is already correct in vLLM from the checkpoint load, so we merge + broadcast just those tensors.
        """
        import math

        cfg = peft_model.peft_config["default"]
        scaling = cfg.lora_alpha / (math.sqrt(cfg.r) if getattr(cfg, "use_rslora", False) else cfg.r)
        adapter = "default"
        sd = self.model.model.state_dict()
        rank0 = torch.distributed.get_rank() == 0
        device = torch.cuda.current_device()

        def clean(name):
            # strip FSDP / activation-checkpoint / PEFT wrapper prefixes -> base HF param name
            for tok in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "base_model.model."):
                name = name.replace(tok, "")
            return name

        def to_vllm(train_name):
            # text-only training names (model.*) -> multimodal checkpoint names (language_model.model.*)
            return ("language_model." + train_name) if train_name.startswith("model.") else train_name

        def full(t):
            t = t.to(device, non_blocking=True)
            return t.full_tensor() if isinstance(t, DTensor) else t

        target_bases = [k for k in sd if k.endswith(".base_layer.weight")]
        if rank0 and not getattr(self, "_merge_logged", False):
            self._merge_logged = True
            sample = target_bases[0] if target_bases else None
            tn = clean(sample) if sample else None
            print(
                f"[lora-merge] {len(target_bases)} target weights; scaling={scaling:.4f}; "
                f"sample {sample} -> vllm={to_vllm(tn) if tn else None}",
                flush=True,
            )

        from torch.multiprocessing.reductions import reduce_tensor

        world_size = torch.distributed.get_world_size()

        def compute_merged(base_key, a_key, b_key):
            # Gather full (collective across all training ranks) then merge on the full tensors.
            # NOTE: full() is a collective — it must run on ALL ranks, never inside an `if rank0`
            # guard, or the collective desyncs and NCCL aborts. Norms are computed here (all ranks)
            # and only logged on rank 0.
            Wf = full(sd[base_key]).float()
            delta = scaling * (full(sd[b_key]).float() @ full(sd[a_key]).float())
            merged = (Wf + delta).to(generator_dtype)
            return merged, Wf.norm().item(), delta.norm().item()

        for base_key in target_bases:
            prefix = base_key[: -len("base_layer.weight")]
            a_key = f"{prefix}lora_A.{adapter}.weight"
            b_key = f"{prefix}lora_B.{adapter}.weight"
            if a_key not in sd or b_key not in sd:
                if rank0:
                    print(f"[lora-merge] missing lora A/B for {base_key}; skipping", flush=True)
                continue
            # Keep the ".base_layer.weight" suffix: vLLM is ALSO LoRA-wrapped, so its params_dict
            # keys are e.g. "...gate_proj.base_layer.weight". vLLM's stacked mapping then rewrites
            # gate_proj->gate_up_proj / q_proj->qkv_proj on this name and finds the fused base_layer.
            vllm_name = to_vllm(clean(base_key))

            if self.use_cuda_ipc:
                # Colocated mode: a training rank and a vLLM engine share a physical GPU, so an NCCL
                # broadcast group would see a duplicate GPU. Share the merged tensor via CUDA IPC
                # handles instead (same mechanism the non-LoRA colocated path uses). Await the send so
                # the merged tensor stays alive until vLLM has reconstructed it from the handle.
                merged, wnorm, dnorm = compute_merged(base_key, a_key, b_key)
                merged = merged.detach().contiguous()
                if rank0 and base_key == target_bases[0] and getattr(self, "_delta_logged", 0) < 5:
                    self._delta_logged = getattr(self, "_delta_logged", 0) + 1
                    print(
                        f"[lora-merge] delta check {clean(base_key)}: |W|={wnorm:.3f} "
                        f"|delta|={dnorm:.4f} |merged|={merged.float().norm().item():.3f}",
                        flush=True,
                    )
                ipc_handle = {get_physical_gpu_id(): reduce_tensor(merged)}
                handle_list = [None] * world_size
                torch.distributed.all_gather_object(handle_list, ipc_handle)
                if torch.distributed.get_rank() == 0:
                    handles = {}
                    for d in handle_list:
                        handles.update(d)
                    await inference_engine_client.update_named_weights(
                        {
                            "names": [vllm_name],
                            "dtypes": [self.cfg.generator.model_dtype],
                            "shapes": [list(merged.shape)],
                            "extras": [{"ipc_handles": handles}],
                        }
                    )
                torch.distributed.barrier()
                torch.cuda.synchronize()
                del merged
                if torch.distributed.get_rank() == 0:
                    torch.cuda.ipc_collect()
            else:
                # Non-colocated: NCCL broadcast. Send the update RPC as a task and run the gather+
                # broadcast in a thread so the event loop can deliver the RPC concurrently (else the
                # collective deadlocks).
                merged_shape = list(sd[base_key].shape)
                if torch.distributed.get_rank() == 0:
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [vllm_name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [merged_shape],
                            }
                        )
                    )

                def gather_merge_broadcast():
                    merged, _, _ = compute_merged(base_key, a_key, b_key)
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(merged.data, 0, group=self._model_update_group)

                await asyncio.to_thread(gather_merge_broadcast)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()
        if fsdp_version(self.model.model) == 1:
            FSDP.set_state_dict_type(
                self.model.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        # Whether to also sync the co-trained modality projector to the inference engine on the
        # LoRA path. This MUST happen whenever the projector is trainable: otherwise vLLM keeps a
        # stale projector (Proj_0) while FSDP recomputes with the live one (Proj_t), producing a
        # train/rollout logprob divergence that GROWS over training (measured ~30% of the tail
        # growth on the overfit hook run). The previous env-var gate (SKYRL_SYNC_MODALITY_UNDER_LORA)
        # never reliably reached the Ray worker, so the trained projector silently never synced.
        # Gate on ACTUAL trainability (known from the model, robust to env propagation).
        # Kill-switch SKYRL_NO_MODALITY_SYNC=1 disables it if the old single-engine generation hang
        # (documented when projector+LoRA weights were first synced) recurs.
        _mm_gate = getattr(self.model, "modalities_manager", None)
        _projector_trainable = False
        if _mm_gate is not None:
            for _mid, _role, _mod in _mm_gate.iter_handler_modules():
                if isinstance(_mod, torch.nn.Module) and any(p.requires_grad for p in _mod.parameters()):
                    _projector_trainable = True
                    break
        sync_modality_under_lora = (
            _projector_trainable or os.environ.get("SKYRL_SYNC_MODALITY_UNDER_LORA", "0") == "1"
        ) and os.environ.get("SKYRL_NO_MODALITY_SYNC", "0") != "1"
        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # vLLM ignores LoRA on the prompt_embeds path, so instead of syncing an adapter we merge
            # the LoRA delta into the base weights and overwrite vLLM's base model with the result.
            await self._merge_lora_into_base_and_broadcast(peft_model, inference_engine_client, generator_dtype)
            if not sync_modality_under_lora:
                return
            # Fall through to sync the co-trained modality projector too (true on-policy). Base
            # params are covered by the merged broadcast above, so leave `params` empty.
            params = {}
        else:
            # Regular model without LoRA
            params = {
                name: tensor
                for name, tensor in self.model.model.state_dict().items()
                if "_modality_" not in name and not name.startswith("modalities.")
            }

        modality_params = []
        # BUGFIX: modalities_manager lives on the HFModelWrapper (self.model), NOT on the inner
        # FSDP-wrapped HF module. Reading it off self.model.model (or its _fsdp_wrapped_module)
        # always returned None, so the co-trained projector was silently never collected/synced to
        # vLLM -> rollouts kept the stale SFT projector -> flat reward on the overfit sanity test.
        modalities_manager = getattr(self.model, "modalities_manager", None)
        if modalities_manager is not None:
            for modality_id, role, module in modalities_manager.iter_handler_modules():
                if not isinstance(module, torch.nn.Module):
                    continue
                for sub_name, sub_param in module.named_parameters():
                    modality_params.append((f"modalities.{modality_id}.{role}.{sub_name}", sub_param))

        if not self.use_cuda_ipc:
            for name, param in params.items():
                if torch.distributed.get_rank() == 0:
                    shape = param.shape

                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [shape],
                            }
                        )
                    )

                # broadcast
                def gather_and_broadcast(param):
                    # For FSDP, gather parameter and broadcast to all InferenceEngines by rank 0
                    device = torch.cuda.current_device()
                    param = param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param
                    # cast to generator dtype
                    param = param.to(generator_dtype)
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self._model_update_group)

                await asyncio.to_thread(gather_and_broadcast, param)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()

            for name, param in modality_params:
                tensor = param.data.to(generator_dtype)
                if torch.distributed.get_rank() == 0:
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [list(param.shape)],
                            }
                        )
                    )

                await asyncio.to_thread(torch.distributed.broadcast, tensor, 0, self._model_update_group)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
            torch.distributed.barrier()
        # CUDA IPC
        else:
            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
            current_size = 0

            module_to_params: Dict[str, List[str]] = {}
            for param_name, param in params.items():
                # TODO (sumanthrh): When would this fail? Works for many AutoModelForCausalLM models for now
                module_name = ".".join(param_name.split(".")[:-2])
                if module_name not in module_to_params:
                    module_to_params[module_name] = [param_name]
                else:
                    module_to_params[module_name].append(param_name)

            # NOTE (sumanthrh): We sync weights module by module. Ex: weights for self attn together, weights for mlp together
            # For FlashRL integration, we allocate new storage for each param. Since q, k and v layer weights are fused internally by vllm,
            # we need to pass the weights for all of these together.
            # Overall, this doesn't hurt perf even in the general case

            for module_name, param_names in module_to_params.items():
                for i, name in enumerate(param_names):
                    param = params[name]
                    module_done = i == len(param_names) - 1

                    from torch.multiprocessing.reductions import reduce_tensor

                    device = torch.cuda.current_device()
                    param = param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param
                    param = param.to(generator_dtype)
                    weight = param.detach().contiguous()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        current_size += weight.nbytes
                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                        weights_update_request["shapes"].append(param.shape)
                        weights_update_request["extras"].append({"ipc_handles": ipc_handles})
                        # We send in batches as an optimization
                        # sync if threshold is reached
                        if (
                            module_done
                            and current_size / (1024**3) > self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB
                        ):
                            await inference_engine_client.update_named_weights(weights_update_request)

                            current_size = 0
                            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
                            # force collect any sent tensors if possible to be memory efficient
                            torch.cuda.ipc_collect()
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

            # sync any remaining weights
            if len(weights_update_request["names"]) > 0 and torch.distributed.get_rank() == 0:
                await asyncio.create_task(inference_engine_client.update_named_weights(weights_update_request))
                torch.cuda.ipc_collect()
            torch.distributed.barrier()
            torch.cuda.synchronize()

            if modality_params:
                from torch.multiprocessing.reductions import reduce_tensor

                weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
                for name, param in modality_params:
                    # BUGFIX: projector params are FSDP-ignored (replicated full tensor on every
                    # rank), so each rank must build an IPC handle for its OWN GPU and we all-gather
                    # them, exactly like the base-weight path above. Previously only rank-0's GPU
                    # handle was sent, so the other engines KeyError'd (vllm_engine.py:180) / kept
                    # the stale projector -> at most 1/N rollouts saw the trained projector.
                    tensor = param.data.to(generator_dtype).contiguous()
                    ipc_handle = {get_physical_gpu_id(): reduce_tensor(tensor)}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)
                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)
                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                        weights_update_request["shapes"].append(list(param.shape))
                        weights_update_request["extras"].append({"ipc_handles": ipc_handles})
                if torch.distributed.get_rank() == 0 and weights_update_request["names"]:
                    await asyncio.create_task(inference_engine_client.update_named_weights(weights_update_request))
                    torch.cuda.ipc_collect()
            torch.distributed.barrier()
            torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.trainer.critic.model.lora.rank,
                lora_alpha=self.cfg.trainer.critic.model.lora.alpha,
                lora_dropout=self.cfg.trainer.critic.model.lora.dropout,
                target_modules=self.cfg.trainer.target_modules,
                value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
                init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
                sequence_parallel_size=self.cfg.trainer.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                modalities_config=self.cfg.trainer.modalities,
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.trainer.gradient_checkpointing:
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.ref.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                bf16=self.cfg.trainer.bf16,
                sequence_parallel_size=self.cfg.trainer.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                modalities_config=self.cfg.trainer.modalities,
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)
