import os
import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Optional
from transformers import AutoModel
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from data_collator import SelfDistillationDataCollator


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


class EMAUpdateCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class BaseSnapshotUpdateCallback(TrainerCallback):


    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.trainer.use_hidden_penalty:
            return

        if not self.trainer.accelerator.sync_gradients:
            return

        current_step = int(state.global_step)

        if (
            current_step in self.trainer.base_snapshot_steps
            and current_step
            != self.trainer._last_base_snapshot_step
        ):
            self.trainer._update_base_snapshot(
                current_step=current_step,
            )

        if (
            current_step
            in self.trainer.base_teacher_snapshot_steps
            and current_step
            != self.trainer._last_base_teacher_snapshot_step
        ):
            self.trainer._update_base_teacher_snapshot(
                current_step=current_step,
            )


class GOLDVLLMSyncCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class OPSDTrainer(SFTTrainer):
    _tag_names = ["trl", "opsd"]
    _name = "OPSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        fixed_teacher: bool = False,
        reason_first: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        penalty_weight: float = 0.0,
        use_hidden_penalty: bool = False,
        hidden_penalty_weight: float = 0.0,
        base_snapshot_steps: tuple[int, ...] = (
            25,
            50,
            75,
            100,
            125,
            150,
            175,
        ),
        base_teacher_snapshot_steps: tuple[int, ...] = (
            25,
            50,
            75,
            100,
            125,
            150,
            175,
        ),
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                reason_first=reason_first,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None
        self.penalty_weight = penalty_weight
        self.use_hidden_penalty = use_hidden_penalty
        self.hidden_penalty_weight = hidden_penalty_weight

        self.base_snapshot_steps = tuple(
            sorted(
                set(
                    int(step)
                    for step in base_snapshot_steps
                )
            )
        )

        self.base_teacher_snapshot_steps = tuple(
            sorted(
                set(
                    int(step)
                    for step in base_teacher_snapshot_steps
                )
            )
        )

        self.student_adapter_name = "default"

        self.base_snapshot_adapter_name = "base_snapshot"

        self.base_teacher_snapshot_adapter_name = (
            "base_teacher_snapshot"
        )

        self._last_base_snapshot_step = 0
        self._active_base_snapshot_step = 0

        self._last_base_teacher_snapshot_step = 0
        self._active_base_teacher_snapshot_step = 0

        if self.use_hidden_penalty:
            if not is_peft_model(self.model):
                raise ValueError(
                    "动态 Base snapshot 当前要求使用 PEFT/LoRA。"
                    "请在启动参数中加入 --use_peft。"
                )

            unwrapped_model = self.accelerator.unwrap_model(
                self.model
            )

            if (
                self.student_adapter_name
                not in unwrapped_model.peft_config
            ):
                raise ValueError(
                    "找不到 Student 的 default adapter。"
                )

            student_adapter_config = (
                unwrapped_model.peft_config[
                    self.student_adapter_name
                ]
            )

            base_snapshot_config = copy.deepcopy(
                student_adapter_config
            )

            unwrapped_model.add_adapter(
                self.base_snapshot_adapter_name,
                base_snapshot_config,
            )

            base_teacher_snapshot_config = copy.deepcopy(
                student_adapter_config
            )

            unwrapped_model.add_adapter(
                self.base_teacher_snapshot_adapter_name,
                base_teacher_snapshot_config,
            )

            self._copy_adapter_weights(
                source_adapter=self.student_adapter_name,
                target_adapter=self.base_snapshot_adapter_name,
            )

            self._copy_adapter_weights(
                source_adapter=self.student_adapter_name,
                target_adapter=(
                    self.base_teacher_snapshot_adapter_name
                ),
            )

            self._freeze_snapshot_adapters()

            unwrapped_model.set_adapter(
                self.student_adapter_name
            )

            self.add_callback(
                BaseSnapshotUpdateCallback(self)
            )

            print(f"\n{'=' * 80}")
            print("DYNAMIC BASE SNAPSHOTS ENABLED")
            print(
                "Base snapshot steps: "
                f"{self.base_snapshot_steps}"
            )
            print(
                "Base Teacher snapshot steps: "
                f"{self.base_teacher_snapshot_steps}"
            )
            print(
                "Base input: student prompt + completion"
            )
            print(
                "Base Teacher input: "
                "teacher/reference prompt + completion"
            )
            print(f"{'=' * 80}\n")

        if self.use_hidden_penalty:
            print(f"\n{'=' * 80}")
            print("CUSTOM HIDDEN PENALTY ENABLED")
            print(f"{'=' * 80}\n")

        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'=' * 80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'=' * 80}\n")

        if self.fixed_teacher:
            print(f"\n{'=' * 80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'=' * 80}\n")

        if self.reason_first:
            print(f"\n{'=' * 80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'=' * 80}\n")

        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if top_k is not None and top_k > 0:
                _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            jsd = beta * kl_teacher + (1 - beta) * kl_student

        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _update_ema(self):

        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):

        if self._ema_params is None:
            yield
            return

        unwrapped = self.accelerator.unwrap_model(model)

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def _freeze_adapter(
        self,
        adapter_name: str,
    ):


        unwrapped_model = self.accelerator.unwrap_model(
            self.model
        )

        marker = f".{adapter_name}."

        for name, param in unwrapped_model.named_parameters():
            if marker in name:
                param.requires_grad_(False)

    def _freeze_base_snapshot_adapter(self):


        self._freeze_adapter(
            self.base_snapshot_adapter_name
        )

    def _freeze_base_teacher_snapshot_adapter(self):


        self._freeze_adapter(
            self.base_teacher_snapshot_adapter_name
        )

    def _freeze_snapshot_adapters(self):

        self._freeze_base_snapshot_adapter()
        self._freeze_base_teacher_snapshot_adapter()

    def _copy_adapter_weights(
        self,
        source_adapter: str,
        target_adapter: str,
    ):


        unwrapped_model = self.accelerator.unwrap_model(
            self.model
        )

        source_marker = f".{source_adapter}."
        target_marker = f".{target_adapter}."

        source_parameters = {}
        target_parameters = {}

        for name, parameter in unwrapped_model.named_parameters():
            if source_marker in name:
                canonical_name = name.replace(
                    source_marker,
                    ".<adapter>.",
                    1,
                )
                source_parameters[canonical_name] = parameter

            elif target_marker in name:
                canonical_name = name.replace(
                    target_marker,
                    ".<adapter>.",
                    1,
                )
                target_parameters[canonical_name] = parameter

        missing_in_source = (
            set(target_parameters) - set(source_parameters)
        )
        missing_in_target = (
            set(source_parameters) - set(target_parameters)
        )

        if missing_in_source or missing_in_target:
            raise RuntimeError(
                "Student/Base snapshot adapter 参数无法对齐。"
                f"missing_in_source={sorted(missing_in_source)[:5]}, "
                f"missing_in_target={sorted(missing_in_target)[:5]}"
            )

        parameter_pairs = [
            (
                source_parameters[name],
                target_parameters[name],
            )
            for name in sorted(source_parameters)
        ]

        zero3_partitioned = any(
            hasattr(parameter, "ds_id")
            for pair in parameter_pairs
            for parameter in pair
        )

        if zero3_partitioned:
            import deepspeed

            gathered_parameters = [
                parameter
                for pair in parameter_pairs
                for parameter in pair
            ]

            with deepspeed.zero.GatheredParameters(
                gathered_parameters,
                modifier_rank=0,
            ):
                with torch.no_grad():
                    for source, target in parameter_pairs:
                        if source.shape != target.shape:
                            raise RuntimeError(
                                "Adapter 参数 shape 不一致："
                                f"source={tuple(source.shape)}, "
                                f"target={tuple(target.shape)}"
                            )

                        target.copy_(
                            source.to(
                                device=target.device,
                                dtype=target.dtype,
                            )
                        )
        else:
            with torch.no_grad():
                for source, target in parameter_pairs:
                    if source.shape != target.shape:
                        raise RuntimeError(
                            "Adapter 参数 shape 不一致："
                            f"source={tuple(source.shape)}, "
                            f"target={tuple(target.shape)}"
                        )

                    target.copy_(
                        source.to(
                            device=target.device,
                            dtype=target.dtype,
                        )
                    )

        if target_adapter in {
            self.base_snapshot_adapter_name,
            self.base_teacher_snapshot_adapter_name,
        }:
            self._freeze_adapter(target_adapter)

    def _update_base_snapshot(self, current_step: int):


        if current_step not in self.base_snapshot_steps:
            return

        self._copy_adapter_weights(
            source_adapter=self.student_adapter_name,
            target_adapter=self.base_snapshot_adapter_name,
        )

        self._last_base_snapshot_step = current_step
        self._active_base_snapshot_step = current_step

        if self.accelerator.is_main_process:
            print(f"\n{'=' * 80}")
            print(
                "BASE SNAPSHOT UPDATED: "
                f"Student step {current_step}"
            )
            print(
                f"Future hidden penalties will use the "
                f"step-{current_step} Student snapshot."
            )
            print(f"{'=' * 80}\n")

    def _update_base_teacher_snapshot(
        self,
        current_step: int,
    ):


        if (
            current_step
            not in self.base_teacher_snapshot_steps
        ):
            return

        self._copy_adapter_weights(
            source_adapter=self.student_adapter_name,
            target_adapter=(
                self.base_teacher_snapshot_adapter_name
            ),
        )

        self._last_base_teacher_snapshot_step = (
            current_step
        )

        self._active_base_teacher_snapshot_step = (
            current_step
        )

        if self.accelerator.is_main_process:
            print(f"\n{'=' * 80}")
            print(
                "BASE TEACHER SNAPSHOT UPDATED: "
                f"Student step {current_step}"
            )
            print(
                "Future Base Teacher hidden penalties "
                "will use the snapshot from "
                f"step {current_step}."
            )
            print(f"{'=' * 80}\n")

    def _forward_snapshot_hidden(
        self,
        model,
        adapter_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ):


        unwrapped_model = self.accelerator.unwrap_model(
            model
        )

        unwrapped_model.set_adapter(adapter_name)
        self._freeze_adapter(adapter_name)

        try:
            with torch.no_grad():
                causal_lm = unwrapped_model.get_base_model()

                base_model_prefix = getattr(
                    causal_lm,
                    "base_model_prefix",
                    "model",
                )

                backbone = getattr(
                    causal_lm,
                    base_model_prefix,
                    None,
                )

                if backbone is None:
                    raise RuntimeError(
                        "无法找到 Transformer backbone："
                        f"prefix={base_model_prefix}, "
                        f"adapter={adapter_name}"
                    )

                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )

                last_hidden = outputs.last_hidden_state[
                    :, prompt_length:, :
                ].detach()

                del outputs

            return last_hidden

        finally:
            unwrapped_model.set_adapter(
                self.student_adapter_name
            )

            self._freeze_snapshot_adapters()

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
        student_hidden_1 = student_last_hidden.float() - student_last_hidden.float().mean(dim=1, keepdim=True)
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
        base_teacher_hidden_1 = base_teacher_last_hidden.detach().float() - base_teacher_last_hidden.detach().float().mean(
            dim=1, keepdim=True)
        hidden_mask = token_mask.unsqueeze(-1).to(
            dtype=student_hidden.dtype
        )

        student_hidden = student_hidden * hidden_mask
        student_hidden_1 = student_hidden_1 * hidden_mask
        base_hidden = base_hidden * hidden_mask
        base_teacher_hidden = (
                base_teacher_hidden * hidden_mask
        )
        base_teacher_hidden_1 = base_teacher_hidden_1 * hidden_mask
        B, T, H = student_hidden_1.shape
        base_teacher_hidden_1_flat = base_teacher_hidden_1.reshape(B * T, H)
        student_hidden_1_flat = student_hidden_1.detach().float().reshape(B * T, H)
        cross_cov = (base_teacher_hidden_1_flat.T) @ (student_hidden_1_flat)
        U, _, Vh = torch.linalg.svd(
            cross_cov,
            full_matrices=False,
        )
        rotation = U @ Vh
        base_teacher_hidden_1 = base_teacher_hidden_1 @ rotation

        student_gram = torch.bmm(
            student_hidden,
            student_hidden.transpose(1, 2),
        )
        student_gram_1 = torch.bmm(
            student_hidden_1.transpose(1, 2),
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
            base_teacher_hidden_1.transpose(1, 2),
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
        gram_difference_2 = student_gram - base_teacher_gram
        gram_difference_3 = student_gram_1 - base_teacher_gram_1
        print("student_gram_1:", student_gram_1.pow(2).sum())
        print("base_teacher_gram_1", base_teacher_gram_1.pow(2).sum())

        squared_gram_difference_1 = (
                gram_difference_1.pow(2) * pair_mask_float
        )
        squared_gram_difference_2 = (
                gram_difference_2.pow(2) * pair_mask_float
        )
        gram_mse_loss_3 = (gram_difference_3 / (
                    base_teacher_gram_1.size(0) * base_teacher_gram_1.size(-1) * base_teacher_gram_1.size(-1))).pow(
            2).sum()
        gram_mse_loss_1 = (
                squared_gram_difference_1.sum()
                / valid_pair_count.clamp_min(1.0)
        )
        gram_mse_loss_2 = (
                squared_gram_difference_2.sum()
                / valid_pair_count.clamp_min(1.0)
        )

        hidden_loss = gram_mse_loss_1 + gram_mse_loss_2
        print("gram_mse_loss_1 is ", gram_mse_loss_1)
        print("gram_mse_loss_2 is ", gram_mse_loss_2)
        print("gram_mse_loss_3 is ", gram_mse_loss_3)
        la1 = 1
        la2 = 0.5
        print("la1:", la1)
        print("la2:", la2)
        return self.hidden_penalty_weight * (la1 * gram_mse_loss_3 + la2 * gram_mse_loss_2)
    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
    ):


        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]

        sampled_token_ids = inputs["student_input_ids"][
            :, student_prompt_len:
        ]

        shifted_labels = inputs["labels"][
            :, student_prompt_len:
        ]

        need_hidden_states = (
            self.use_hidden_penalty
            and self.hidden_penalty_weight != 0
        )

        if need_hidden_states:
            unwrapped_model = self.accelerator.unwrap_model(
                model
            )
            unwrapped_model.set_adapter(
                self.student_adapter_name
            )
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )

        student_logits = outputs_student.logits[
            :, student_prompt_len - 1: -1, :
        ]

        if need_hidden_states:
            student_last_hidden = outputs_student.hidden_states[-1][
                :, student_prompt_len:, :
            ]
        else:
            student_last_hidden = None

        if self.use_thinking_machines_loss:
            student_log_probs = F.log_softmax(
                student_logits / self.temperature,
                dim=-1,
            )

            student_log_probs_sampled = torch.gather(
                student_log_probs,
                dim=-1,
                index=sampled_token_ids.unsqueeze(-1),
            ).squeeze(-1)

            del student_log_probs
            del student_logits
        else:
            student_logits_for_loss = student_logits

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        if self.use_ema_teacher:
            adapter_context = self._ema_teacher_context(model)
        elif self.fixed_teacher and is_peft_model(model):
            adapter_context = (
                self.accelerator.unwrap_model(model).disable_adapter()
            )
        else:
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

            teacher_logits = outputs_teacher.logits[
                :, teacher_prompt_len - 1: -1, :
            ]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(
                    teacher_logits / self.temperature,
                    dim=-1,
                )

                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs,
                    dim=-1,
                    index=sampled_token_ids.unsqueeze(-1),
                ).squeeze(-1)

                del teacher_log_probs
                del teacher_logits
            else:
                teacher_logits_for_loss = teacher_logits

            del outputs_teacher
            empty_cache()

        if need_hidden_states:
            base_last_hidden = (
                self._forward_snapshot_hidden(
                    model=model,
                    adapter_name=(
                        self.base_snapshot_adapter_name
                    ),
                    input_ids=inputs[
                        "student_input_ids"
                    ],
                    attention_mask=inputs[
                        "student_attention_mask"
                    ],
                    prompt_length=student_prompt_len,
                )
            )

            base_teacher_last_hidden = (
                self._forward_snapshot_hidden(
                    model=model,
                    adapter_name=(
                        self.base_teacher_snapshot_adapter_name
                    ),
                    input_ids=inputs[
                        "teacher_input_ids"
                    ],
                    attention_mask=inputs[
                        "teacher_attention_mask"
                    ],
                    prompt_length=teacher_prompt_len,
                )
            )

            if (
                base_last_hidden.shape[1]
                != base_teacher_last_hidden.shape[1]
            ):
                raise RuntimeError(
                    "Base 与 Base Teacher 的 completion "
                    "hidden 长度不一致。"
                    f"base={tuple(base_last_hidden.shape)}, "
                    "base_teacher="
                    f"{tuple(base_teacher_last_hidden.shape)}, "
                    f"student_prompt_len={student_prompt_len}, "
                    f"teacher_prompt_len={teacher_prompt_len}"
                )

        if self.use_thinking_machines_loss:
            advantage = (
                teacher_log_probs_sampled
                - student_log_probs_sampled
            ).detach()

            if shifted_labels is not None:
                mask = shifted_labels != -100

                valid_advantage = advantage[mask]
                valid_student_log_probs = student_log_probs_sampled[mask]
            else:
                valid_advantage = advantage
                valid_student_log_probs = student_log_probs_sampled

            if valid_student_log_probs.numel() > 0:
                loss = -(
                    valid_advantage * valid_student_log_probs
                ).mean()
            else:
                loss = student_log_probs_sampled.sum() * 0.0

            del student_log_probs_sampled
            del teacher_log_probs_sampled
            del advantage
            del valid_advantage
            del valid_student_log_probs

        else:
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )

            del student_logits_for_loss
            del teacher_logits_for_loss

        if need_hidden_states:
            custom_penalty = self._compute_custom_hidden_penalty(
                student_last_hidden=student_last_hidden,
                base_last_hidden=base_last_hidden,
                base_teacher_last_hidden=(
                    base_teacher_last_hidden
                ),
                shifted_labels=shifted_labels,
            )

            loss = loss + custom_penalty

            mode = "train" if model.training else "eval"
            self._metrics[mode]["custom_hidden_penalty"].append(
                custom_penalty.detach().float().item()
            )

            del student_last_hidden
            del base_last_hidden
            del base_teacher_last_hidden
            del custom_penalty

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output

        return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning about the solution."""
        if self.use_vllm:
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            with torch.no_grad():
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=teacher_reasoning_prompts,
                            attention_mask=teacher_reasoning_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts only."""
        import time

        start_time = time.time()

        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'=' * 80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Batch size: {inputs['student_prompts'].shape[0]}")
        print(f"  Prompt length: {inputs['student_prompts'].shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'=' * 80}\n")

        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            generated_tokens = generated_outputs.sequences
        finally:
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts using vLLM."""
        import time

        device = self.accelerator.device

        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                padded_completion_ids_list.append(completion_tensor)

        padded_completion_ids = torch.stack(padded_completion_ids_list)

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning using vLLM."""
        import time

        device = self.accelerator.device

        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                if self.is_fsdp_enabled:
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    for name, param in self.model.named_parameters():
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                self.model.unmerge_adapter()
        else:
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 80}")
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'=' * 80}\n")

        self._generation_outputs_buffer.clear()

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:

        on_policy = True

        if self.reason_first:
            print(f"\n{'=' * 80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'=' * 80}\n")

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                if random.random() < 0.01:
                    print(f"\n{'=' * 80}")
                    print(f"TEACHER REASONING SAMPLE (Step {self.state.global_step}):")
                    print(f"{'=' * 80}")
                    sample_idx = random.randint(0, len(reasoning_texts) - 1)
                    print(f"\n{'=' * 80}")
                    sample_prompt = self.processing_class.decode(
                        inputs["teacher_reasoning_prompts"][sample_idx], skip_special_tokens=False
                    )
                    print(f"PROMPT:\n{sample_prompt}")
                    print(f"\nReasoning:\n{reasoning_texts[sample_idx]}")
                    print(f"{'=' * 80}\n")

                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        student_prompt_len = inputs["student_prompt_length"]

        generation_ids = generated_ids[:, student_prompt_len:]

        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        if random.random() < 0.01:
            print(f"\n{'=' * 80}")
            print(f"STUDENT GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"{'=' * 80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"{'=' * 80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)

        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})