
from contextlib import contextmanager, nullcontext
import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, _sd_cfg
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.import_utils import deprecated
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (
    calculate_workload,
    ceildiv,
    get_seqlen_balanced_partitions,
    prepare_dynamic_batch,
    restore_dynamic_batch,
    roundup_divisible)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@deprecated("legacy worker implementation is deprecated and will be removed in v0.8.0")
class DataParallelPPOActor(BasePPOActor):


    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        self.use_hidden_penalty = bool(
            _sd_cfg(config, "use_hidden_penalty", False)
        )

        self.hidden_penalty_weight = float(
            _sd_cfg(config, "hidden_penalty_weight", 0.0)
        )

        self.hidden_snapshot_interval = int(
            _sd_cfg(config, "hidden_snapshot_interval", 25)
        )

        self.hidden_snapshot_last_step = int(
            _sd_cfg(config, "hidden_snapshot_last_step", 175)
        )
        if self.use_hidden_penalty and self.use_fused_kernels:
            raise ValueError(
                "use_hidden_penalty=True is not supported with "
                "use_fused_kernels=True because the fused output contract "
                "does not guarantee hidden_states."
            )
        self._hidden_snapshot_step = 0
        self._hidden_snapshot_source_step = 0
        self._hidden_snapshot_shadow = None
        if (
            self.use_hidden_penalty
            and actor_optimizer is not None
        ):
            self._hidden_snapshot_shadow = {}

            for name, param in self.actor_module.named_parameters():
                if param.requires_grad:
                    self._hidden_snapshot_shadow[name] = (
                        param.data.detach().clone()
                    )

            if torch.distributed.get_rank() == 0:
                snapshot_mb = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in self._hidden_snapshot_shadow.values()
                ) / 1e6

                print(
                    "Hidden snapshot initialized: "
                    f"params={len(self._hidden_snapshot_shadow)}, "
                    f"size={snapshot_mb:.1f}MB"
                )
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))

        # Self-distillation: teacher shadow weights (EMA / snapshot)
        self._teacher_shadow = None
        self._teacher_mode = _sd_cfg(config, "teacher_mode", "fixed")
        self._teacher_ema_decay = float(_sd_cfg(config, "ema_decay", 0.95))
        self._teacher_sync_interval = int(_sd_cfg(config, "teacher_sync_interval", 10))
        self._teacher_step_counter = 0
        if self._teacher_mode in ("ema", "snapshot") and actor_optimizer is not None:
            # Initialize shadow from current trainable params
            self._teacher_shadow = {}
            for name, param in self.actor_module.named_parameters():
                if param.requires_grad:
                    self._teacher_shadow[name] = param.data.detach().clone()
            if torch.distributed.get_rank() == 0:
                n_shadow = len(self._teacher_shadow)
                shadow_mb = sum(p.numel() * p.element_size() for p in self._teacher_shadow.values()) / 1e6
                print(f"Teacher shadow initialized: mode={self._teacher_mode}, "
                      f"params={n_shadow}, size={shadow_mb:.1f}MB")
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: int | None = None,
        topk_indices: torch.Tensor | None = None,

        align_response_by_mask: bool = False,
        return_last_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        return_topk_indices = use_topk and topk_indices is None
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
                and not return_last_hidden
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False))

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_mask = micro_batch.get("response_mask")
            align_response_by_mask = align_response_by_mask and response_mask is not None
            entropy = None
            last_hidden = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype)
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype)
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size)
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_last_hidden,
                    return_dict=True,
                    **extra_args)  # prevent model thinks we are generating
                last_hidden_rmpad = None
                if return_last_hidden:
                    if (
                        not hasattr(output, "hidden_states")
                        or output.hidden_states is None
                    ):
                        raise RuntimeError(
                            "return_last_hidden=True, but the model did not "
                            "return hidden_states."
                        )

                    last_hidden_rmpad = output.hidden_states[-1]
                    if last_hidden_rmpad.ndim == 3:
                        if last_hidden_rmpad.shape[0] != 1:
                            raise RuntimeError(
                                "Unexpected remove-padding hidden-state shape: "
                                f"{tuple(last_hidden_rmpad.shape)}"
                            )
                        last_hidden_rmpad = last_hidden_rmpad.squeeze(0)

                    if last_hidden_rmpad.ndim != 2:
                        raise RuntimeError(
                            "Expected compact hidden states with shape "
                            "[tokens, hidden_size], but got "
                            f"{tuple(last_hidden_rmpad.shape)}."
                        )
                if self.use_fused_kernels:
                    if return_all_logps or use_topk:
                        raise ValueError("full_logit_distill/top-k distillation is not supported with fused kernels enabled.")
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_log_probs_rmpad = None
                    if return_all_logps:
                        all_log_probs_rmpad = F.log_softmax(logits_rmpad, dim=-1)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            _, topk_indices_rmpad = self._chunked_rowwise_topk(logits_rmpad, topk)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = self._build_full_topk_indices(
                                topk_indices=topk_indices,
                                batch_size=batch_size,
                                seqlen=seqlen,
                                response_length=response_length,
                                response_mask=response_mask if align_response_by_mask else None,
                                attention_mask=attention_mask if align_response_by_mask else None)
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"),
                                indices)
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0),
                                    dim=1,
                                    padding=True).squeeze(0)
                        log_probs, topk_log_probs_rmpad = self._chunked_selected_log_probs(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            topk_indices=topk_indices_rmpad)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward)

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if return_all_logps:
                        all_log_probs_rmpad = gather_outputs_and_unpad(
                            all_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if use_topk:
                        topk_log_probs_rmpad = gather_outputs_and_unpad(
                            topk_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                    if return_last_hidden:
                        last_hidden_rmpad = gather_outputs_and_unpad(
                            last_hidden_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if return_all_logps:
                        all_log_probs_rmpad = all_log_probs_rmpad[:0]
                    if use_topk:
                        topk_log_probs_rmpad = topk_log_probs_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]
                    if return_last_hidden:
                        last_hidden_rmpad = last_hidden_rmpad[:0]
 
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen)
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen)
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen)
                if return_all_logps:
                    full_all_log_probs = pad_input(
                        hidden_states=all_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen)
                if use_topk:
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen)
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
                if return_last_hidden:
                    full_last_hidden = pad_input(
                        hidden_states=last_hidden_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if align_response_by_mask:
                    response_lengths = response_mask.sum(dim=1).to(dtype=torch.long)
                    log_probs = self._align_compact_response_tensor(
                        full_log_probs.squeeze(-1),
                        response_length=response_length,
                        response_lengths=response_lengths,
                        attention_mask=attention_mask,
                    )
                    if calculate_entropy:
                        entropy = self._align_compact_response_tensor(
                            full_entropy.squeeze(-1),
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared = self._align_compact_response_tensor(
                            full_sum_pi_squared.squeeze(-1),
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                    if return_all_logps:
                        all_log_probs = self._align_compact_response_tensor(
                            full_all_log_probs,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                    if use_topk:
                        topk_log_probs = self._align_compact_response_tensor(
                            full_topk_log_probs,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                        if return_topk_indices:
                            topk_indices = self._align_compact_response_tensor(
                                full_topk_indices,
                                response_length=response_length,
                                response_lengths=response_lengths,
                                attention_mask=attention_mask,
                            )
                    if return_last_hidden:
                        last_hidden = self._align_compact_response_hidden(
                            hidden=full_last_hidden,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                else:
                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if calculate_sum_pi_squared:
                        # (bsz, response_length)
                        sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if return_all_logps:
                        all_log_probs = full_all_log_probs[:, -response_length - 1 : -1, :]
                    if use_topk:
                        topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]
                        if return_topk_indices:
                            topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

                    if return_last_hidden:
                        if response_length == 0:
                            last_hidden = full_last_hidden[:, 0:0, :]
                        else:
                            # Log-prob positions predict response tokens:
                            # [-response_length - 1 : -1].
                            # Hidden states represent response tokens themselves:
                            # [-response_length:].
                            last_hidden = full_last_hidden[
                                :, -response_length:, :
                            ]
            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_last_hidden,
                    return_dict=True,
                    **extra_args)  # prevent model thinks we are generating
                full_last_hidden = None
                if return_last_hidden:
                    if (
                        not hasattr(output, "hidden_states")
                        or output.hidden_states is None
                    ):
                        raise RuntimeError(
                            "return_last_hidden=True, but the model did not "
                            "return hidden_states."
                        )

                    full_last_hidden = output.hidden_states[-1]
                    if full_last_hidden.ndim != 3:
                        raise RuntimeError(
                            "Expected full hidden states with shape "
                            "[batch, sequence, hidden_size], but got "
                            f"{tuple(full_last_hidden.shape)}."
                        )
                if self.use_fused_kernels:
                    if return_all_logps or use_topk:
                        raise ValueError("full_logit_distill/top-k distillation is not supported with fused kernels enabled.")
                    if align_response_by_mask:
                        response_lengths = response_mask.sum(dim=1).to(dtype=torch.long)
                        log_probs = self._align_compact_response_tensor(
                            output.log_probs,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                        entropy = self._align_compact_response_tensor(
                            output.entropy,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                    else:
                        log_probs = output.log_probs[:, -response_length - 1 : -1]
                        entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    if align_response_by_mask:
                        response_lengths = response_mask.sum(dim=1).to(dtype=torch.long)
                        logits = self._align_compact_response_tensor(
                            logits,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                    else:
                        logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if return_all_logps:
                        all_log_probs = F.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            _, topk_indices = self._chunked_rowwise_topk(logits, topk)
                        log_probs, topk_log_probs = self._chunked_selected_log_probs(
                            logits=logits,
                            labels=micro_batch["responses"],
                            topk_indices=topk_indices)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )
                if return_last_hidden:
                    if align_response_by_mask:
                        response_lengths = response_mask.sum(
                            dim=1
                        ).to(dtype=torch.long)

                        last_hidden = self._align_compact_response_hidden(
                            hidden=full_last_hidden,
                            response_length=response_length,
                            response_lengths=response_lengths,
                            attention_mask=attention_mask,
                        )
                    elif response_length == 0:
                        last_hidden = full_last_hidden[:, 0:0, :]
                    else:
                        last_hidden = full_last_hidden[
                            :, -response_length:, :
                        ]
            outputs = {"log_probs": log_probs}
            if return_last_hidden:
                if last_hidden is None:
                    raise RuntimeError(
                        "return_last_hidden=True, but response-aligned hidden "
                        "states were not produced."
                    )
                outputs["last_hidden"] = last_hidden
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if return_all_logps:
                outputs["all_log_probs"] = all_log_probs
            if use_topk:
                outputs["topk_log_probs"] = topk_log_probs
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs
    def _align_compact_response_hidden(
        self,
        hidden: torch.Tensor,
        response_length: int,
        response_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        predictor_starts = self._response_predictor_starts(
            attention_mask=attention_mask,
            response_lengths=response_lengths,
        )

        hidden_starts = predictor_starts + 1

        aligned = hidden.new_zeros(
            hidden.shape[0],
            response_length,
            hidden.shape[-1],
         )

        for row_idx, (row_len, hidden_start) in enumerate(
            zip(
                response_lengths.tolist(),
                hidden_starts.tolist(),
                strict=True,
            )
        ):
            if row_len <= 0:
                continue

            if row_len > response_length:
                raise RuntimeError(
                    "Response length exceeds the compact response width: "
                    f"row={row_idx}, row_length={row_len}, "
                    f"response_length={response_length}."
                )

            hidden_end = hidden_start + row_len
            if hidden_start < 0 or hidden_end > hidden.shape[1]:
                raise RuntimeError(
                    "Hidden-state alignment exceeded the attended sequence: "
                    f"row={row_idx}, hidden_start={hidden_start}, "
                    f"hidden_end={hidden_end}, "
                    f"sequence_length={hidden.shape[1]}."
                )

            aligned[row_idx, :row_len, :] = hidden[
                row_idx, hidden_start:hidden_end, :
            ]

        return aligned
    def _build_hidden_shifted_labels(
            self,
            responses: torch.Tensor,
            response_mask: torch.Tensor,
        ) -> torch.Tensor:
            if responses.ndim != 2:
                raise ValueError(

                )
    
            if tuple(response_mask.shape) != tuple(responses.shape):
                raise ValueError(

                )
    
            valid_mask = response_mask.to(
                device=responses.device,
                dtype=torch.bool,
            )
    
            shifted_labels = responses.to(dtype=torch.long).clone()
            shifted_labels.masked_fill_(~valid_mask, -100)
    
            return shifted_labels
    def _build_full_topk_indices(
        self,
        topk_indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        response_length: int,
        response_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Expand response-space top-k indices into full-sequence positions."""
        topk = topk_indices.size(-1)
        full_topk_indices = torch.zeros(
            batch_size,
            seqlen,
            topk,
            device=topk_indices.device,
            dtype=topk_indices.dtype)
        if response_mask is None:
            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
            return full_topk_indices

        assert attention_mask is not None, "attention_mask is required to align compact response tensors."
        response_lengths = response_mask.sum(dim=1).to(dtype=torch.long)
        predictor_starts = self._response_predictor_starts(attention_mask, response_lengths)
        for row_idx, (row_len, predictor_start) in enumerate(
            zip(response_lengths.tolist(), predictor_starts.tolist(), strict=True)
        ):
            if row_len <= 0:
                continue
            full_topk_indices[row_idx, predictor_start : predictor_start + row_len, :] = topk_indices[
                row_idx, :row_len, :
            ]
        return full_topk_indices

    def _response_predictor_starts(
        self,
        attention_mask: torch.Tensor,
        response_lengths: torch.Tensor) -> torch.Tensor:
        """Locate each row's first response predictor from its actual attended span."""
        sequence_positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        last_attended = torch.where(
            attention_mask.to(dtype=torch.bool),
            sequence_positions.unsqueeze(0),
            sequence_positions.new_full((1,), -1),
        ).amax(dim=1)
        predictor_starts = last_attended - response_lengths
        nonempty_rows = response_lengths > 0
        assert torch.all(predictor_starts[nonempty_rows] >= 0), (
            "Teacher inputs must contain at least one attended prompt token before every non-empty response."
        )
        return predictor_starts

    def _align_compact_response_tensor(
        self,
        tensor: torch.Tensor,
        response_length: int,
        response_lengths: torch.Tensor,
        attention_mask: torch.Tensor) -> torch.Tensor:
        """Map compact teacher-response tensors back to fixed student response length."""
        predictor_starts = self._response_predictor_starts(attention_mask, response_lengths)
        aligned_shape = (tensor.shape[0], response_length, *tensor.shape[2:])
        aligned = tensor.new_zeros(aligned_shape)
        for row_idx, (row_len, predictor_start) in enumerate(
            zip(response_lengths.tolist(), predictor_starts.tolist(), strict=True)
        ):
            if row_len <= 0:
                continue
            aligned[row_idx, :row_len, ...] = tensor[
                row_idx, predictor_start : predictor_start + row_len, ...
            ]
        return aligned

    def _chunked_rowwise_topk(self, logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute row-wise top-k in small chunks to limit workspace size on long sequences."""
        chunk_rows = max(int(self.config.get("distill_chunk_rows", 128)), 1)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        if flat_logits.numel() == 0:
            empty_shape = (*logits.shape[:-1], topk)
            empty_values = logits.new_empty(empty_shape)
            empty_indices = torch.empty(empty_shape, device=logits.device, dtype=torch.long)
            return empty_values, empty_indices

        topk_values = []
        topk_indices = []
        for chunk_logits in flat_logits.split(chunk_rows, dim=0):
            chunk_values, chunk_indices = torch.topk(chunk_logits, topk, dim=-1)
            topk_values.append(chunk_values)
            topk_indices.append(chunk_indices)

        out_shape = (*logits.shape[:-1], topk)
        return torch.cat(topk_values, dim=0).view(out_shape), torch.cat(topk_indices, dim=0).view(out_shape)

    def _chunked_selected_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action log-probs and aligned top-k log-probs with shared chunked logsumexp."""
        chunk_rows = max(int(self.config.get("distill_chunk_rows", 128)), 1)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_topk_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
        if flat_logits.numel() == 0:
            empty_log_probs = logits.new_empty(labels.shape)
            empty_topk = logits.new_empty(topk_indices.shape)
            return empty_log_probs, empty_topk

        log_prob_chunks = []
        topk_log_prob_chunks = []
        for chunk_start in range(0, flat_logits.shape[0], chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, flat_logits.shape[0])
            chunk_logits = flat_logits[chunk_start:chunk_end]
            chunk_labels = flat_labels[chunk_start:chunk_end]
            chunk_topk_indices = flat_topk_indices[chunk_start:chunk_end]

            chunk_logsumexp = torch.logsumexp(chunk_logits.float(), dim=-1, keepdim=True)
            chunk_label_logits = torch.gather(chunk_logits, dim=-1, index=chunk_labels.unsqueeze(-1)).float()
            chunk_topk_logits = torch.gather(chunk_logits, dim=-1, index=chunk_topk_indices).float()

            log_prob_chunks.append((chunk_label_logits - chunk_logsumexp).squeeze(-1).to(chunk_logits.dtype))
            topk_log_prob_chunks.append((chunk_topk_logits - chunk_logsumexp).to(chunk_logits.dtype))

        return (
            torch.cat(log_prob_chunks, dim=0).view(labels.shape),
            torch.cat(topk_log_prob_chunks, dim=0).view(topk_indices.shape))

# ---- Teacher shadow weight management ----
    @contextmanager
    def _hidden_snapshot_context(self):

        if self._hidden_snapshot_shadow is None:
            yield None
            return

        backup = {}

        for name, param in self.actor_module.named_parameters():
            if name not in self._hidden_snapshot_shadow:
                continue

            backup[name] = param.data.detach().clone()

            snapshot_param = self._hidden_snapshot_shadow[name]
            if (
                snapshot_param.device != param.data.device
                or snapshot_param.dtype != param.data.dtype
            ):
                snapshot_param = snapshot_param.to(
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                self._hidden_snapshot_shadow[name] = snapshot_param

            param.data.copy_(snapshot_param)

        try:
            yield True
        finally:
            for name, param in self.actor_module.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

    @torch.no_grad()
    def _refresh_hidden_snapshot(self) -> bool:
        if not self._hidden_snapshot_shadow:
            return False

        copied_params = 0

        for name, param in self.actor_module.named_parameters():
            if name not in self._hidden_snapshot_shadow:
                continue

            snapshot_param = self._hidden_snapshot_shadow[name]

            if (
                snapshot_param.device != param.data.device
                or snapshot_param.dtype != param.data.dtype
            ):
                snapshot_param = snapshot_param.to(
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                self._hidden_snapshot_shadow[name] = snapshot_param

            snapshot_param.copy_(param.data.detach())
            copied_params += 1

        return copied_params > 0
    def _update_hidden_snapshot(
        self,
        global_step: int,
        optimizer_updated: bool,
    ) -> bool:
        if not self.use_hidden_penalty:
            return False

        if self.hidden_penalty_weight == 0:
            return False

        if not optimizer_updated:
            return False

        interval = int(self.hidden_snapshot_interval)
        last_step = int(self.hidden_snapshot_last_step)

        if interval <= 0:
            raise ValueError("")

        if global_step <= 0:
            return False
        if last_step > 0 and global_step > last_step:
            return False

        if global_step % interval != 0:
            return False

        if global_step <= self._hidden_snapshot_source_step:
            return False

        refreshed = self._refresh_hidden_snapshot()

        if not refreshed:
            raise RuntimeError(
            )

        self._hidden_snapshot_source_step = global_step
        self._hidden_snapshot_step = global_step

        return True
    def _update_teacher_shadow(self):
        """Update teacher shadow weights after optimizer step."""
        if self._teacher_shadow is None:
            return
        self._teacher_step_counter += 1
        if self._teacher_mode == "ema":
            decay = self._teacher_ema_decay
            for name, param in self.actor_module.named_parameters():
                if name in self._teacher_shadow:
                    self._teacher_shadow[name].mul_(decay).add_(param.data.detach(), alpha=1 - decay)
        elif self._teacher_mode == "snapshot":
            if self._teacher_step_counter % self._teacher_sync_interval == 0:
                for name, param in self.actor_module.named_parameters():
                    if name in self._teacher_shadow:
                        self._teacher_shadow[name].copy_(param.data.detach())

    @contextmanager
    def _teacher_forward_context(self):
        """Temporarily swap the actor into teacher mode for one or more forwards."""
        if self._teacher_mode in ("ema", "snapshot"):
            if self._teacher_shadow is None:
                yield None
                return
            backup = {}
            for name, param in self.actor_module.named_parameters():
                if name in self._teacher_shadow:
                    backup[name] = param.data.detach().clone()
                    param.data.copy_(self._teacher_shadow[name])
            adapter_ctx = nullcontext()
        elif self._teacher_mode == "fixed" and hasattr(self.actor_module, "disable_adapter"):
            backup = None
            adapter_ctx = self.actor_module.disable_adapter()
        else:
            backup = None
            adapter_ctx = nullcontext()

        try:
            with adapter_ctx:
                yield True
        finally:
            if backup is not None:
                for name, param in self.actor_module.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name])

    @torch.no_grad()
    def _teacher_forward(
        self,
        model_inputs,
        temperature,
        calculate_entropy=True,
        return_all_logps=False,
        distill_topk=None,
        topk_indices=None,
        align_response_by_mask=False):
        with torch.no_grad():
            with self._teacher_forward_context() as teacher_ready:
                if teacher_ready is None:
                    return None
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_all_logps=return_all_logps,
                    distill_topk=distill_topk,
                    topk_indices=topk_indices,
                    align_response_by_mask=align_response_by_mask)

        return outputs

    @torch.no_grad()
    def _teacher_forward_multi(self, model_inputs, temperature, calculate_entropy=True):
        valid_mask = model_inputs["valid_mask"].to(dtype=torch.bool)
        responses = model_inputs["responses"]
        response_mask = model_inputs["response_mask"]
        batch_size, num_ctx = valid_mask.shape
        response_length = responses.shape[1]

        log_probs = torch.zeros(
            (batch_size, num_ctx, response_length),
            device=responses.device,
            dtype=torch.float32)
        entropys = (
            torch.zeros((batch_size, num_ctx, response_length), device=responses.device, dtype=torch.float32)
            if calculate_entropy
            else None
        )

        if batch_size == 0 or num_ctx == 0:
            outputs = {"log_probs": log_probs, "valid_mask": valid_mask}
            if entropys is not None:
                outputs["entropys"] = entropys
            return outputs

        with torch.no_grad():
            with self._teacher_forward_context() as teacher_ready:
                if teacher_ready is None:
                    return None
                for ctx_idx in range(num_ctx):
                    ctx_outputs = self._forward_micro_batch(
                        {
                            "responses": responses,
                            "response_mask": response_mask,
                            "input_ids": model_inputs["input_ids"][:, ctx_idx, ...],
                            "attention_mask": model_inputs["attention_mask"][:, ctx_idx, ...],
                            "position_ids": model_inputs["position_ids"][:, ctx_idx, ...]},
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        align_response_by_mask=True)

                    ctx_valid = valid_mask[:, ctx_idx].to(dtype=log_probs.dtype).unsqueeze(-1)
                    log_probs[:, ctx_idx, :] = ctx_outputs["log_probs"].to(dtype=log_probs.dtype) * ctx_valid
                    if entropys is not None and "entropys" in ctx_outputs:
                        entropys[:, ctx_idx, :] = (
                            ctx_outputs["entropys"].to(dtype=entropys.dtype) * ctx_valid.to(dtype=entropys.dtype)
                        )

        outputs = {"log_probs": log_probs, "valid_mask": valid_mask}
        if entropys is not None:
            outputs["entropys"] = entropys
        return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.actor_module.parameters(),
                max_norm=self.config.grad_clip,
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(),
                max_norm=self.config.grad_clip,
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()


        if self.scaler is not None:
            old_scale = self.scaler.get_scale()

            self.scaler.step(self.actor_optimizer)
            self.scaler.update()

            new_scale = self.scaler.get_scale()
            optimizer_updated = new_scale >= old_scale
        else:
            if not torch.isfinite(grad_norm):
                optimizer_updated = False
                print(
                    f"WARN: rank {torch.distributed.get_rank()} "
                    f"grad_norm is not finite: {grad_norm}"
                )
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
                optimizer_updated = True
        if (
            optimizer_updated
            and getattr(self.actor_module, "_qat_fuse_enabled", False)
        ):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm, optimizer_updated
    def _prepare_update_micro_batches(
        self,
        mini_batch: DataProto,
        needs_teacher_forward: bool,
        needs_hidden_penalty: bool,
    ) -> list[DataProto]:
        if not self.config.use_dynamic_bsz:
            self.gradient_accumulation = (
                self.config.ppo_mini_batch_size
                // self.config.ppo_micro_batch_size_per_gpu
            )
            return mini_batch.split(
                self.config.ppo_micro_batch_size_per_gpu
            )

        max_token_len = (
            self.config.ppo_max_token_len_per_gpu
            * self.ulysses_sequence_parallel_size
        )
        dp_group = torch.distributed.group.WORLD

        if not needs_teacher_forward:
            micro_batches, _ = prepare_dynamic_batch(
                mini_batch,
                max_token_len=max_token_len,
                dp_group=dp_group,
            )
            return micro_batches

        teacher_masks = [
            mini_batch.batch[key]
            for key in (
                "teacher_attention_mask",
                "teacher_correct_attention_mask",
                "teacher_wrong_attention_mask",
                "teacher_correct_multi_attention_mask",
                "teacher_wrong_multi_attention_mask",
            )
            if key in mini_batch.batch.keys()
        ]

        if not teacher_masks:
            micro_batches, _ = prepare_dynamic_batch(
                mini_batch,
                max_token_len=max_token_len,
                dp_group=dp_group,
            )
            return micro_batches

        student_mask = mini_batch.batch["attention_mask"]
        student_effective_seq_lens = (
            student_mask.sum(dim=1).to(dtype=torch.long)
        )
        student_workloads = calculate_workload(
            student_effective_seq_lens
        ).to(dtype=torch.long)

        effective_seq_lens = student_effective_seq_lens
        effective_workloads = student_workloads.clone()
        for teacher_mask in teacher_masks:
            if teacher_mask.dim() == 3:
                teacher_effective_seq_lens = (
                    teacher_mask.sum(dim=-1).to(dtype=torch.long)
                )

                effective_seq_lens = torch.maximum(
                    effective_seq_lens,
                    teacher_effective_seq_lens.amax(dim=1),
                )

                # Grouped teacher contexts are forwarded slot-by-slot.
                effective_workloads = (
                    effective_workloads
                    + calculate_workload(
                        teacher_effective_seq_lens
                    ).sum(dim=1)
                )
            else:
                teacher_effective_seq_lens = (
                    teacher_mask.sum(dim=1).to(dtype=torch.long)
                )

                effective_seq_lens = torch.maximum(
                    effective_seq_lens,
                    teacher_effective_seq_lens,
                )

                effective_workloads = (
                    effective_workloads
                    + calculate_workload(
                        teacher_effective_seq_lens
                    )
                )

        if needs_hidden_penalty:
            if "teacher_attention_mask" not in mini_batch.batch:
                raise RuntimeError(
                    "Hidden penalty requires teacher_attention_mask, "
                    "but it is missing from the mini-batch."
                )

            hidden_teacher_mask = mini_batch.batch[
                "teacher_attention_mask"
            ]

            if hidden_teacher_mask.dim() != 2:
                raise RuntimeError(
                    "teacher_attention_mask used by the hidden snapshot "
                    "must have shape [batch, sequence], but received "
                    f"{tuple(hidden_teacher_mask.shape)}."
                )

            hidden_teacher_effective_seq_lens = (
                hidden_teacher_mask.sum(dim=1).to(dtype=torch.long)
            )
            hidden_teacher_workloads = calculate_workload(
                hidden_teacher_effective_seq_lens
            ).to(dtype=torch.long)

            effective_workloads = (
                effective_workloads
                + student_workloads
                + hidden_teacher_workloads
            )

            effective_seq_lens = torch.maximum(
                effective_seq_lens,
                hidden_teacher_effective_seq_lens,
            )

        batch_idx_list = (
            self._get_batch_idx_list_from_effective_workloads(
                effective_seq_lens=effective_seq_lens,
                effective_workloads=effective_workloads,
                max_token_len=max_token_len,
                dp_group=dp_group,
            )
        )

        return [
            mini_batch.select_idxs(batch_idx)
            for batch_idx in batch_idx_list
        ]
    def _get_batch_idx_list_from_effective_workloads(
        self,
        effective_seq_lens: torch.Tensor,
        effective_workloads: torch.Tensor,
        max_token_len: int,
        dp_group) -> list[list[int]]:
        """Mirror dynamic batching using the total per-sample update workload."""
        batch_size = int(effective_seq_lens.numel())
        if batch_size == 0:
            return []

        max_effective_seq_len = int(effective_seq_lens.max().item())
        assert max_token_len >= max_effective_seq_len, (
            f"max_token_len must be greater than the effective sequence length. "
            f"Got max_token_len={max_token_len} and max_effective_seq_len={max_effective_seq_len}"
        )

        max_workload = int(
            calculate_workload(
                torch.tensor([max_token_len], device=effective_workloads.device, dtype=torch.long)
            ).item()
        )
        total_workload = int(effective_workloads.sum().item())
        num_micro_batches = min(batch_size, ceildiv(total_workload, max(max_workload, 1)))
        if torch.distributed.is_initialized() and dp_group is not None:
            num_micro_batches_tensor = torch.tensor([num_micro_batches], device=get_device_name())
            torch.distributed.all_reduce(num_micro_batches_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_group)
            num_micro_batches = int(num_micro_batches_tensor.cpu().item())
        if getattr(self, "ulysses_sequence_parallel_size", 1) > 1:
            num_micro_batches = roundup_divisible(num_micro_batches, self.ulysses_sequence_parallel_size)
        num_micro_batches = min(num_micro_batches, batch_size)

        workloads = effective_workloads.long().cpu().tolist()
        batch_idx_list = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)
        batch_idx_list.sort(
            key=lambda partition: (sum(workloads[idx] for idx in partition), partition[0] if partition else 0),
            reverse=True)
        batch_idx_list = batch_idx_list[::2][::-1] + batch_idx_list[1::2]
        return batch_idx_list

    def _get_batch_idx_list_from_effective_lengths(
        self,
        effective_seq_lens: torch.Tensor,
        max_token_len: int,
        dp_group) -> list[list[int]]:
        batch_size = int(effective_seq_lens.numel())
        if batch_size == 0:
            return []

        max_effective_seq_len = int(effective_seq_lens.max().item())
        assert max_token_len >= max_effective_seq_len, (
            f"max_token_len must be greater than the effective sequence length. "
            f"Got max_token_len={max_token_len} and max_effective_seq_len={max_effective_seq_len}"
        )

        total_seqlen = int(effective_seq_lens.sum().item())
        num_micro_batches = min(batch_size, ceildiv(total_seqlen, max_token_len))
        if torch.distributed.is_initialized() and dp_group is not None:
            num_micro_batches_tensor = torch.tensor([num_micro_batches], device=get_device_name())
            torch.distributed.all_reduce(num_micro_batches_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_group)
            num_micro_batches = int(num_micro_batches_tensor.cpu().item())
        if getattr(self, "ulysses_sequence_parallel_size", 1) > 1:
            num_micro_batches = roundup_divisible(num_micro_batches, self.ulysses_sequence_parallel_size)
        num_micro_batches = min(num_micro_batches, batch_size)

        workloads = calculate_workload(effective_seq_lens.long()).cpu().tolist()
        batch_idx_list = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)
        batch_idx_list.sort(
            key=lambda partition: (sum(workloads[idx] for idx in partition), partition[0] if partition else 0),
            reverse=True)
        batch_idx_list = batch_idx_list[::2][::-1] + batch_idx_list[1::2]
        return batch_idx_list

    def _get_teacher_forward_batch_idx_list(self, attention_mask: torch.Tensor) -> list[list[int]]:
        """Split teacher-only forwards into smaller chunks to cap peak memory."""
        batch_size = int(attention_mask.shape[0])
        if batch_size == 0:
            return []

        if self.config.use_dynamic_bsz:
            effective_seq_lens = attention_mask.sum(dim=1).to(dtype=torch.long)
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            return self._get_batch_idx_list_from_effective_lengths(
                effective_seq_lens=effective_seq_lens,
                max_token_len=max_token_len,
                dp_group=torch.distributed.group.WORLD)

        chunk_size = self.config.ppo_micro_batch_size_per_gpu or batch_size
        chunk_size = max(int(chunk_size), 1)
        return [list(range(start, min(start + chunk_size, batch_size))) for start in range(0, batch_size, chunk_size)]

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len, dp_group=torch.distributed.group.WORLD
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        global_steps = data.meta_info.get("global_steps")

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Self-distillation: include teacher data if present
        if "teacher_log_probs" in data.batch.keys():
            select_keys.append("teacher_log_probs")
        if "teacher_entropy" in data.batch.keys():
            select_keys.append("teacher_entropy")
        for teacher_key in ("teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"):
            if teacher_key in data.batch.keys():
                select_keys.append(teacher_key)
        for teacher_prefix in ("teacher_correct", "teacher_wrong"):
            for teacher_suffix in ("input_ids", "attention_mask", "position_ids"):
                teacher_key = f"{teacher_prefix}_{teacher_suffix}"
                if teacher_key in data.batch.keys():
                    select_keys.append(teacher_key)
        for teacher_prefix in ("teacher_correct_multi", "teacher_wrong_multi"):
            for teacher_suffix in ("input_ids", "attention_mask", "position_ids", "valid_mask"):
                teacher_key = f"{teacher_prefix}_{teacher_suffix}"
                if teacher_key in data.batch.keys():
                    select_keys.append(teacher_key)
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if "global_steps" not in data.meta_info:
            raise KeyError(
            )
        
        global_step = int(data.meta_info["global_steps"])
        
        hidden_snapshot_used_step = self._hidden_snapshot_source_step
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0}
        # 必须使用与 trainer 日志一致的 global_steps
        if "global_steps" not in data.meta_info:
            raise KeyError(
            )
        
        global_step = int(data.meta_info["global_steps"])
        
        hidden_snapshot_used_step = self._hidden_snapshot_source_step
        optimizer_updated_in_this_call = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                mini_batch_loss_mode = self.config.policy_loss.get(
                    "loss_mode",
                    "vanilla",
                )

                needs_teacher_forward = mini_batch_loss_mode in (
                    "opsd",
                    "opsd_ectr",
                    "sdpo",
                    "srpo",
                    "rlsd",
                    "rlsd_ectr",
                    "rlcsd",
                ) and any(
                    key in mini_batch.batch.keys()
                    for key in (
                        "teacher_input_ids",
                        "teacher_attention_mask",
                        "teacher_position_ids",
                        "teacher_correct_input_ids",
                        "teacher_correct_attention_mask",
                        "teacher_correct_position_ids",
                        "teacher_wrong_input_ids",
                        "teacher_wrong_attention_mask",
                        "teacher_wrong_position_ids",
                        "teacher_correct_multi_input_ids",
                        "teacher_correct_multi_attention_mask",
                        "teacher_correct_multi_position_ids",
                        "teacher_correct_multi_valid_mask",
                        "teacher_wrong_multi_input_ids",
                        "teacher_wrong_multi_attention_mask",
                        "teacher_wrong_multi_position_ids",
                        "teacher_wrong_multi_valid_mask",
                    )
                )

                needs_hidden_penalty = (
                    self.use_hidden_penalty
                    and self.hidden_penalty_weight != 0
                    and mini_batch_loss_mode in (
                        "sdpo",
                        "srpo",
                        "rlsd",
                    )
                )

                micro_batches = self._prepare_update_micro_batches(
                    mini_batch,
                    needs_teacher_forward=needs_teacher_forward,
                    needs_hidden_penalty=needs_hidden_penalty,
                )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    need_teacher = loss_mode in ("opsd", "opsd_ectr", "sdpo", "rlsd", "rlsd_ectr", "srpo","rlcsd")
                    need_full_distill = need_teacher and loss_mode in ("opsd", "opsd_ectr", "sdpo", "srpo") and _sd_cfg(
                        self.config, "full_logit_distill", True
                    )
                    top_k_distill = _sd_cfg(self.config, "top_k_distill", 0) if need_full_distill else 0
                    top_k_distill = int(top_k_distill or 0)
                    use_sparse_topk = need_full_distill and top_k_distill > 0

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_lp = model_inputs.get("teacher_log_probs")
                    teacher_ent = model_inputs.get("teacher_entropy")
                    teacher_wrong_lp = model_inputs.get("teacher_wrong_log_probs")
                    teacher_wrong_ent = model_inputs.get("teacher_wrong_entropy")
                    teacher_correct_multi_lp = model_inputs.get("teacher_correct_multi_log_probs")
                    teacher_wrong_multi_lp = model_inputs.get("teacher_wrong_multi_log_probs")
                    teacher_correct_multi_ent = model_inputs.get("teacher_correct_multi_entropy")
                    teacher_wrong_multi_ent = model_inputs.get("teacher_wrong_multi_entropy")
                    teacher_correct_multi_valid_mask = model_inputs.get("teacher_correct_multi_valid_mask")
                    teacher_wrong_multi_valid_mask = model_inputs.get("teacher_wrong_multi_valid_mask")
                    teacher_all_log_probs = None
                    teacher_topk_log_probs = None
                    teacher_wrong_topk_log_probs = None
                    teacher_topk_indices = None
                    student_all_log_probs = None
                    student_topk_log_probs = None
                    student_topk_indices = None
                    teacher_outputs = None
                    teacher_has_privileged_inputs = all(
                        key in model_inputs for key in ("teacher_input_ids", "teacher_attention_mask", "teacher_position_ids")
                    )
                    teacher_inputs = None
                    if teacher_has_privileged_inputs:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "response_mask": model_inputs["response_mask"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"]}
                    teacher_correct_inputs = None
                    if all(
                        key in model_inputs
                        for key in (
                            "teacher_correct_input_ids",
                            "teacher_correct_attention_mask",
                            "teacher_correct_position_ids")
                    ):
                        teacher_correct_inputs = {
                            "responses": model_inputs["responses"],
                            "response_mask": model_inputs["response_mask"],
                            "input_ids": model_inputs["teacher_correct_input_ids"],
                            "attention_mask": model_inputs["teacher_correct_attention_mask"],
                            "position_ids": model_inputs["teacher_correct_position_ids"]}
                    teacher_wrong_inputs = None
                    if all(
                        key in model_inputs
                        for key in (
                            "teacher_wrong_input_ids",
                            "teacher_wrong_attention_mask",
                            "teacher_wrong_position_ids")
                    ):
                        teacher_wrong_inputs = {
                            "responses": model_inputs["responses"],
                            "response_mask": model_inputs["response_mask"],
                            "input_ids": model_inputs["teacher_wrong_input_ids"],
                            "attention_mask": model_inputs["teacher_wrong_attention_mask"],
                            "position_ids": model_inputs["teacher_wrong_position_ids"]}
                    teacher_correct_multi_inputs = None
                    if all(
                        key in model_inputs
                        for key in (
                            "teacher_correct_multi_input_ids",
                            "teacher_correct_multi_attention_mask",
                            "teacher_correct_multi_position_ids",
                            "teacher_correct_multi_valid_mask")
                    ):
                        teacher_correct_multi_inputs = {
                            "responses": model_inputs["responses"],
                            "response_mask": model_inputs["response_mask"],
                            "input_ids": model_inputs["teacher_correct_multi_input_ids"],
                            "attention_mask": model_inputs["teacher_correct_multi_attention_mask"],
                            "position_ids": model_inputs["teacher_correct_multi_position_ids"],
                            "valid_mask": model_inputs["teacher_correct_multi_valid_mask"]}
                    teacher_wrong_multi_inputs = None
                    if all(
                        key in model_inputs
                        for key in (
                            "teacher_wrong_multi_input_ids",
                            "teacher_wrong_multi_attention_mask",
                            "teacher_wrong_multi_position_ids",
                            "teacher_wrong_multi_valid_mask")
                    ):
                        teacher_wrong_multi_inputs = {
                            "responses": model_inputs["responses"],
                            "response_mask": model_inputs["response_mask"],
                            "input_ids": model_inputs["teacher_wrong_multi_input_ids"],
                            "attention_mask": model_inputs["teacher_wrong_multi_attention_mask"],
                            "position_ids": model_inputs["teacher_wrong_multi_position_ids"],
                            "valid_mask": model_inputs["teacher_wrong_multi_valid_mask"]}

                    # OPSD sparse distillation uses teacher-selected top-k support, so run teacher first.
                    if need_teacher and use_sparse_topk and loss_mode in ("opsd", "opsd_ectr"):
                        if loss_mode == "opsd_ectr":
                            assert teacher_correct_inputs is not None, (
                                "opsd_ectr requires teacher_correct_* tensors in the batch."
                            )
                            teacher_forward_inputs = teacher_correct_inputs
                            align_resp = True
                        else:
                            teacher_forward_inputs = teacher_inputs if teacher_inputs is not None else model_inputs
                            align_resp = teacher_inputs is not None
                        teacher_outputs = self._teacher_forward(
                            teacher_forward_inputs,
                            temperature=temperature,
                            calculate_entropy=True,
                            distill_topk=top_k_distill,
                            align_response_by_mask=align_resp)
                        if teacher_outputs is not None:
                            teacher_lp = teacher_outputs["log_probs"]
                            teacher_ent = teacher_outputs.get("entropys")
                            teacher_topk_log_probs = teacher_outputs.get("topk_log_probs")
                            teacher_topk_indices = teacher_outputs.get("topk_indices")
                    need_hidden_penalty = needs_hidden_penalty
                    if need_hidden_penalty and self._hidden_snapshot_shadow is None:
                        raise RuntimeError(
                            "Hidden penalty is enabled, but the hidden snapshot "
                            "was not initialized. Hidden penalty can only be used "
                            "by an actor with a non-None optimizer."
                        )
                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=need_full_distill and not use_sparse_topk,
                        distill_topk=(
                            top_k_distill
                            if use_sparse_topk
                            and loss_mode in ("sdpo", "srpo")
                            else None
                        ),
                        topk_indices=(
                            teacher_topk_indices
                            if use_sparse_topk
                            and loss_mode in ("opsd", "opsd_ectr")
                            else None
                        ),
                        return_last_hidden=need_hidden_penalty,
                    )
                    student_last_hidden = (
                        outputs.get("last_hidden")
                        if need_hidden_penalty
                        else None
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_log_probs = outputs.get("all_log_probs")
                    student_topk_log_probs = outputs.get("topk_log_probs")
                    student_topk_indices = outputs.get("topk_indices")

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    if need_teacher and teacher_outputs is None:
                        if loss_mode == "rlsd_ectr":
                            assert teacher_correct_inputs is not None and teacher_wrong_inputs is not None, (
                                f"{loss_mode} requires teacher_correct_* and teacher_wrong_* tensors in the batch."
                            )
                            teacher_correct_outputs = self._teacher_forward(
                                teacher_correct_inputs,
                                temperature=temperature,
                                calculate_entropy=True,
                                align_response_by_mask=True)
                            teacher_wrong_outputs = self._teacher_forward(
                                teacher_wrong_inputs,
                                temperature=temperature,
                                calculate_entropy=True,
                                align_response_by_mask=True)
                            teacher_lp = teacher_correct_outputs["log_probs"]
                            teacher_ent = teacher_correct_outputs.get("entropys")
                            teacher_wrong_lp = teacher_wrong_outputs["log_probs"]
                            teacher_wrong_ent = teacher_wrong_outputs.get("entropys")
                        elif loss_mode == "rlcsd":
                            assert teacher_correct_inputs is not None and teacher_wrong_multi_inputs is not None, (
                                f"{loss_mode} requires teacher_correct_* and teacher_wrong_multi_* tensors in the batch."
                            )
                            teacher_correct_outputs = self._teacher_forward(
                                teacher_correct_inputs,
                                temperature=temperature,
                                calculate_entropy=True,
                                align_response_by_mask=True)
                            teacher_wrong_multi_outputs_local = self._teacher_forward_multi(
                                teacher_wrong_multi_inputs,
                                temperature=temperature,
                                calculate_entropy=True)
                            teacher_lp = teacher_correct_outputs["log_probs"]
                            teacher_ent = teacher_correct_outputs.get("entropys")
                            teacher_wrong_multi_lp = teacher_wrong_multi_outputs_local["log_probs"]
                            teacher_wrong_multi_ent = teacher_wrong_multi_outputs_local.get("entropys")
                            teacher_wrong_multi_valid_mask = teacher_wrong_multi_outputs_local["valid_mask"]
                        else:
                            teacher_forward_inputs = teacher_inputs if teacher_inputs is not None else model_inputs
                            if teacher_inputs is not None:
                                teacher_outputs = self._teacher_forward(
                                    teacher_forward_inputs,
                                    temperature=temperature,
                                    calculate_entropy=True,
                                    return_all_logps=need_full_distill and not use_sparse_topk,
                                    topk_indices=student_topk_indices if use_sparse_topk and loss_mode in ("sdpo", "srpo") else None,
                                    align_response_by_mask=True)
                            elif teacher_lp is None or need_full_distill or (use_sparse_topk and loss_mode in ("sdpo", "srpo")):
                                if self._teacher_mode == "fixed":
                                    if need_full_distill or use_sparse_topk:
                                        teacher_outputs = self._teacher_forward(
                                            model_inputs,
                                            temperature=temperature,
                                            calculate_entropy=True,
                                            return_all_logps=need_full_distill and not use_sparse_topk,
                                            topk_indices=student_topk_indices if use_sparse_topk and loss_mode in ("sdpo", "srpo") else None,
                                            align_response_by_mask=False)
                                    else:
                                        # Fixed teacher = base model = ref policy
                                        teacher_lp = model_inputs.get("ref_log_prob")
                                        teacher_ent = model_inputs.get("teacher_entropy")
                                elif self._teacher_mode in ("ema", "snapshot") and self._teacher_shadow is not None:
                                    # EMA/snapshot: forward pass with shadow weights
                                    teacher_outputs = self._teacher_forward(
                                        model_inputs,
                                        temperature=temperature,
                                        calculate_entropy=True,
                                        return_all_logps=need_full_distill and not use_sparse_topk,
                                        topk_indices=student_topk_indices if use_sparse_topk and loss_mode in ("sdpo", "srpo") else None,
                                        align_response_by_mask=False)

                            if teacher_outputs is not None:
                                teacher_lp = teacher_outputs["log_probs"]
                                teacher_ent = teacher_outputs.get("entropys")
                                teacher_all_log_probs = teacher_outputs.get("all_log_probs")
                                teacher_topk_log_probs = teacher_outputs.get("topk_log_probs")
                    if need_teacher and loss_mode == "opsd_ectr":
                        assert teacher_wrong_inputs is not None and teacher_topk_indices is not None, (
                            "opsd_ectr requires teacher_wrong_* tensors and pre-computed teacher_topk_indices."
                        )
                        teacher_wrong_outputs = self._teacher_forward(
                            teacher_wrong_inputs,
                            temperature=temperature,
                            calculate_entropy=True,
                            topk_indices=teacher_topk_indices,
                            align_response_by_mask=True)
                        teacher_wrong_lp = teacher_wrong_outputs["log_probs"]
                        teacher_wrong_ent = teacher_wrong_outputs.get("entropys")
                        teacher_wrong_topk_log_probs = teacher_wrong_outputs.get("topk_log_probs")
                    base_last_hidden = None
                    base_teacher_last_hidden = None

                    if need_hidden_penalty:
                        if student_last_hidden is None:
                            raise RuntimeError(
                                "Hidden penalty is enabled, but the student "
                                "forward did not return last_hidden."
                            )

                        if teacher_inputs is None:
                            raise RuntimeError(
                                f"{loss_mode} hidden penalty requires privileged "
                                "teacher_input_ids, teacher_attention_mask and "
                                "teacher_position_ids in the actor batch."
                            )

                        # Same frozen snapshot with the ordinary student context.
                        base_last_hidden = self._hidden_snapshot_forward(
                            model_inputs=model_inputs,
                            temperature=temperature,
                            align_response_by_mask=False,
                        )

                        # Same frozen snapshot with the privileged teacher
                        # context and the same completion.
                        base_teacher_last_hidden = (
                            self._hidden_snapshot_forward(
                                model_inputs=teacher_inputs,
                                temperature=temperature,
                                align_response_by_mask=True,
                            )
                        )

                        if base_last_hidden is None:
                            raise RuntimeError(
                                "The ordinary-context hidden snapshot forward "
                                "returned None."
                            )

                        if base_teacher_last_hidden is None:
                            raise RuntimeError(
                                "The privileged-context hidden snapshot forward "
                                "returned None."
                            )

                        expected_shape = student_last_hidden.shape

                        if base_last_hidden.shape != expected_shape:
                            raise RuntimeError(
                                "Student and ordinary snapshot hidden-state "
                                "shapes differ: "
                                f"student={tuple(expected_shape)}, "
                                f"snapshot={tuple(base_last_hidden.shape)}."
                            )

                        if base_teacher_last_hidden.shape != expected_shape:
                            raise RuntimeError(
                                "Student and privileged snapshot hidden-state "
                                "shapes differ: "
                                f"student={tuple(expected_shape)}, "
                                "privileged_snapshot="
                                f"{tuple(base_teacher_last_hidden.shape)}."
                            )
                    loss_kwargs = dict(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        global_steps=global_steps)
                    if loss_mode in ("opsd", "sdpo", "rlsd", "srpo"):
                        loss_kwargs["teacher_log_probs"] = teacher_lp
                        loss_kwargs["teacher_entropy"] = teacher_ent
                        if need_full_distill:
                            if use_sparse_topk:
                                loss_kwargs["student_topk_log_probs"] = student_topk_log_probs
                                loss_kwargs["teacher_topk_log_probs"] = teacher_topk_log_probs
                            else:
                                loss_kwargs["student_all_log_probs"] = student_all_log_probs
                                loss_kwargs["teacher_all_log_probs"] = teacher_all_log_probs
                    elif loss_mode == "opsd_ectr":
                        loss_kwargs["teacher_log_probs"] = teacher_lp
                        loss_kwargs["teacher_entropy"] = teacher_ent
                        loss_kwargs["student_topk_log_probs"] = student_topk_log_probs
                        loss_kwargs["teacher_topk_log_probs"] = teacher_topk_log_probs
                        loss_kwargs["teacher_wrong_topk_log_probs"] = teacher_wrong_topk_log_probs
                    elif loss_mode == "rlsd_ectr":
                        loss_kwargs["teacher_log_probs"] = teacher_lp
                        loss_kwargs["teacher_wrong_log_probs"] = teacher_wrong_lp
                        loss_kwargs["teacher_entropy"] = teacher_ent
                        loss_kwargs["teacher_wrong_entropy"] = teacher_wrong_ent
                    elif loss_mode == "rlcsd":
                        loss_kwargs["teacher_log_probs"] = teacher_lp
                        loss_kwargs["teacher_entropy"] = teacher_ent
                        loss_kwargs["teacher_wrong_multi_log_probs"] = teacher_wrong_multi_lp
                        loss_kwargs["teacher_wrong_multi_valid_mask"] = teacher_wrong_multi_valid_mask
                        loss_kwargs["teacher_wrong_multi_entropy"] = teacher_wrong_multi_ent
                    pg_loss, pg_metrics = policy_loss_fn(**loss_kwargs)
                    micro_batch_metrics.update(pg_metrics)
                    if need_hidden_penalty:
                        shifted_labels = self._build_hidden_shifted_labels(
                            responses=model_inputs["responses"],
                            response_mask=response_mask,
                        )

                        raw_hidden_penalty = (
                            self._compute_custom_hidden_penalty(
                                student_last_hidden=student_last_hidden,
                                base_last_hidden=base_last_hidden,
                                base_teacher_last_hidden=(
                                    base_teacher_last_hidden
                                ),
                                shifted_labels=shifted_labels,
                            )
                        )

                        if raw_hidden_penalty.ndim != 0:
                            raise RuntimeError(
                                "_compute_custom_hidden_penalty() must return "
                                "a scalar tensor, but received shape "
                                f"{tuple(raw_hidden_penalty.shape)}."
                            )

                        custom_hidden_penalty = (
                            self.hidden_penalty_weight
                            * raw_hidden_penalty
                        )

                        pg_loss = pg_loss + custom_hidden_penalty

                        micro_batch_metrics[
                            "actor/custom_hidden_penalty_raw"
                        ] = raw_hidden_penalty.detach().float().item()

                        micro_batch_metrics[
                            "actor/custom_hidden_penalty"
                        ] = custom_hidden_penalty.detach().float().item()

                        token_mask = shifted_labels.ne(-100)

                        response_token_count = (
                            response_mask.to(dtype=torch.float32)
                            .sum()
                            .clamp(min=1.0)
                        )

                        micro_batch_metrics[
                            "actor/hidden_penalty_token_frac"
                        ] = (
                            token_mask.float().sum()
                            / response_token_count
                        ).detach().item()
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask)
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, optimizer_updated = self._optimizer_step()

                if optimizer_updated:
                    self._update_teacher_shadow()
                    optimizer_updated_in_this_call = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        snapshot_refreshed = self._update_hidden_snapshot(
            global_step=global_step,
            optimizer_updated=optimizer_updated_in_this_call,
        )
        
        snapshot_metrics = {
            "actor/hidden_snapshot_used_step": float(
                hidden_snapshot_used_step
            ),
            "actor/hidden_snapshot_latest_step": float(
                self._hidden_snapshot_source_step
            ),
            "actor/hidden_snapshot_refreshed": float(
                snapshot_refreshed
            ),
            "actor/hidden_snapshot_age": float(
                global_step - hidden_snapshot_used_step
            ),
        }
        is_rank_zero = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        
        if is_rank_zero:
            print(
                f"[HiddenSnapshot] "
                f"global_step={global_step} | "
                f"used_snapshot_step={hidden_snapshot_used_step} | "
                f"latest_snapshot_step={self._hidden_snapshot_source_step} | "
                f"refreshed={snapshot_refreshed}",
                flush=True,
            )
        append_to_dict(metrics, snapshot_metrics)
        return metrics
    @torch.no_grad()
    def _hidden_snapshot_forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        temperature: float,
        align_response_by_mask: bool,
    ) -> torch.Tensor | None:
        was_training = self.actor_module.training

        try:
            self.actor_module.eval()

            with self._hidden_snapshot_context() as ready:
                if ready is None:
                    return None

                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    return_all_logps=False,
                    align_response_by_mask=align_response_by_mask,
                    return_last_hidden=True,
                )
        finally:
            self.actor_module.train(was_training)

        last_hidden = outputs.get("last_hidden")
        if last_hidden is None:
            raise RuntimeError(
                "The hidden snapshot forward did not return last_hidden."
            )

        return last_hidden.detach()
    def _compute_custom_hidden_penalty(
        self,
        student_last_hidden,
        base_last_hidden,
        base_teacher_last_hidden,
        shifted_labels=None,
    ):

        if (
            not self.use_hidden_penalty
            or self.hidden_penalty_weight == 0
        ):
            return student_last_hidden.sum() * 0.0

        hidden_states = {
            "student": student_last_hidden,
            "base": base_last_hidden,
            "base_teacher": base_teacher_last_hidden,
        }

        for hidden_name, hidden in hidden_states.items():
            if hidden.ndim != 3:
                raise ValueError(
                    f"{hidden_name} hidden 必须是 [B, T, H]，"
                    f"当前 shape={tuple(hidden.shape)}"
                )

        batch_size, seq_len, _ = (
            student_last_hidden.shape
        )
        for hidden_name, hidden in (
            ("base", base_last_hidden),
            (
                "base_teacher",
                base_teacher_last_hidden,
            ),
        ):
            if hidden.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"Student 和 {hidden_name} 的 batch/"
                    "completion length 必须相同。"
                    f"student={tuple(student_last_hidden.shape)}, "
                    f"{hidden_name}={tuple(hidden.shape)}"
                )

        if shifted_labels is not None:
            expected_shape = (batch_size, seq_len)

            if tuple(shifted_labels.shape) != expected_shape:
                raise ValueError(
                    "shifted_labels shape 不正确。"
                    f"labels={tuple(shifted_labels.shape)}, "
                    f"expected={expected_shape}"
                )

            token_mask = shifted_labels.ne(-100)
        else:
            token_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=student_last_hidden.device,
            )

        if not token_mask.any():
            return student_last_hidden.sum() * 0.0

        student_hidden = F.normalize(
            student_last_hidden.float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        student_hidden_1=student_last_hidden.float()-student_last_hidden.float().mean(dim=1, keepdim=True)
        base_hidden = F.normalize(
            base_last_hidden.detach().float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        base_teacher_hidden = F.normalize(
            base_teacher_last_hidden.detach().float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        base_teacher_hidden_1=base_teacher_last_hidden.detach().float()-base_teacher_last_hidden.detach().float().mean(dim=1, keepdim=True)
        hidden_mask = token_mask.unsqueeze(-1).to(
            dtype=student_hidden.dtype
        )

        student_hidden = student_hidden * hidden_mask
        student_hidden_1=student_hidden_1*hidden_mask
        base_hidden = base_hidden * hidden_mask
        base_teacher_hidden = (
            base_teacher_hidden * hidden_mask
        )
        base_teacher_hidden_1=base_teacher_hidden_1.float()* hidden_mask
        B,T,H=student_hidden_1.shape
        base_teacher_hidden_1_flat=base_teacher_hidden_1.float().reshape(B*T,H)/(base_teacher_hidden.size(0)*base_teacher_hidden.size(1)*base_teacher_hidden.size(-1))
        student_hidden_1_flat=student_hidden_1.detach().float().reshape(B*T,H)
        cross_cov=(base_teacher_hidden_1_flat.float().T)@(student_hidden_1_flat.float())
        U,_,Vh=torch.linalg.svd(
            cross_cov.float(),
            full_matrices=False,
        )
        rotation = U @ Vh
        rotation=rotation.to(base_teacher_hidden_1_flat.dtype)
        base_teacher_hidden_1=base_teacher_hidden_1.float()@rotation.float()
        student_gram = torch.bmm(
            student_hidden,
            student_hidden.transpose(1, 2),
        )
        student_gram_1 = torch.bmm(
            student_hidden_1.transpose(1, 2)/(base_teacher_hidden.size(0)*base_teacher_hidden.size(1)*base_teacher_hidden.size(-1)),
            student_hidden_1,
        )

        base_gram = torch.bmm(
            base_hidden,
            base_hidden.transpose(1, 2),
        )

        base_teacher_gram = torch.bmm(
            base_teacher_hidden,
            base_teacher_hidden.transpose(1, 2),
        )
        base_teacher_gram_1 = torch.bmm(
            base_teacher_hidden_1.transpose(1, 2)/(base_teacher_hidden.size(0)*base_teacher_hidden.size(1)*base_teacher_hidden.size(-1)),
            base_teacher_hidden_1,
        )

        pair_mask = (
            token_mask.unsqueeze(1)
            & token_mask.unsqueeze(2)
        )

        pair_mask_float = pair_mask.to(
            dtype=student_gram.dtype
        )

        valid_pair_count = pair_mask_float.sum()

        if valid_pair_count.item() == 0:
            return student_last_hidden.sum() * 0.0

        mask = token_mask.to(device=student_hidden_1.device, dtype=student_hidden_1.dtype)
        valid_count = mask.sum()

        if valid_count.item() == 0:
            return student_hidden_1.sum() * 0.0

        gram_difference_1 = student_gram - base_gram
        gram_difference_2=student_gram - base_teacher_gram
        gram_difference_3=student_gram_1-base_teacher_gram_1
        print("student_gram_1:",student_gram_1.pow(2).sum())
        print("base_teacher_gram_1",base_teacher_gram_1.pow(2).sum())
        squared_gram_difference_1 = (
            gram_difference_1.pow(2) * pair_mask_float
        )
        squared_gram_difference_2=(
            gram_difference_2.pow(2) * pair_mask_float
        )
        gram_mse_loss_3= (gram_difference_3).pow(2).sum() 
        gram_mse_loss_1 = (
            squared_gram_difference_1.sum()
            / valid_pair_count.clamp_min(1.0)
        )
        gram_mse_loss_2 = (
            squared_gram_difference_2.sum()
            / valid_pair_count.clamp_min(1.0)
        )

        print("gram_mse_loss_1 is ",gram_mse_loss_1)
        print("gram_mse_loss_2 is ",gram_mse_loss_2)
        print("gram_mse_loss_3 is ",gram_mse_loss_3)
        la1=1
        la2=0.5
        print("la1:",la1)
        print("la2:",la2)
        return (la1*gram_mse_loss_3+la2*gram_mse_loss_2)