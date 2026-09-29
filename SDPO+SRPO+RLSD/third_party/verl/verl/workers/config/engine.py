
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig

from ...utils.profiler import ProfilerConfig
from .model import DiffusionModelConfig, HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "FSDPEngineConfig",
    "McoreEngineConfig",
    "TrainingWorkerConfig",
    "TorchtitanEngineConfig",
    "VeOmniEngineConfig",
    "AutomodelEngineConfig",
    "EngineConfig",
    "EngineRouterReplayConfig",
    "QATEngineConfig",
    "MindSpeedEngineConfig",
]


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class EngineRouterReplayConfig(BaseConfig):


    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class EngineConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "micro_batch_size_per_gpu",
        "infer_max_token_len_per_gpu",
        "infer_micro_batch_size_per_gpu",
        "use_fused_kernels",
        "use_remove_padding",
        "forward_only",
        "param_offload",
    }
    # whether to offload param
    param_offload: bool = False
    # whether to offload optimizer
    optimizer_offload: bool = False
    # whether to offload grad
    grad_offload: bool = False
    # whether the engine is forward only (e.g., ref policy)
    forward_only: bool = False
    # the strategy (backend)
    strategy: str = None
    # model dtype
    dtype: str = "bfloat16"  # ["bfloat16", "float16"]
    # whether to use dynamic bsz
    use_dynamic_bsz: bool = True
    # for training
    max_token_len_per_gpu: int = None
    micro_batch_size_per_gpu: int = None
    # for inference
    infer_max_token_len_per_gpu: int = None
    infer_micro_batch_size_per_gpu: int = None
    # whether use fuse lm head kernel
    use_fused_kernels: bool = False

    use_remove_padding: bool = True

    seed: int = 42

    full_determinism: bool = False
    router_replay: EngineRouterReplayConfig = field(default_factory=EngineRouterReplayConfig)

    def __post_init__(self):
        pass

@dataclass
class QATEngineConfig(BaseConfig):


    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None


@dataclass
class McoreEngineConfig(EngineConfig):
    _mutable_fields = EngineConfig._mutable_fields | {"sequence_parallel"}
    # mcore parallelism
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    dynamic_context_parallel: bool = False
    max_seqlen_per_dp_cp_rank: Optional[int] = None
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    dist_checkpointing_prefix: str = ""
    dist_ckpt_optim_fully_reshardable: bool = False
    distrib_optim_fully_reshardable_mem_efficient: bool = False
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    override_mcore_model_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = True
    vanilla_mbridge: bool = True
    strategy: str = "megatron"
    qat: QATEngineConfig = field(default_factory=QATEngineConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        """config validation logics go here"""
        assert self.strategy == "megatron"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class FSDPEngineConfig(EngineConfig):
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_sequence_parallel_size"}

    # fsdp specific flags
    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: Optional[dict[str, Any]] = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "fsdp"
    qat: QATEngineConfig = field(default_factory=QATEngineConfig)

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["fsdp", "fsdp2"], f"strategy {self.strategy} not supported"


@dataclass
class VeOmniEngineConfig(EngineConfig):


    _mutable_fields = EngineConfig._mutable_fields | {"attn_implementation"}

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    forward_prefetch: bool = False
    use_orig_params: bool = False
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "veomni"
    fsdp_size: int = -1
    ulysses_parallel_size: int = 1
    expert_parallel_size: int = 1
    seed: int = 42
    full_determinism: bool = False
    mixed_precision: bool = False
    init_device: str = "meta"
    enable_full_shard: bool = False
    ckpt_manager: Literal["dcp"] = "dcp"
    load_checkpoint_path: Optional[str] = None
    enable_fsdp_offload: bool = False
    enable_reentrant: bool = False
    attn_implementation: str = "flash_attention_2"
    moe_implementation: str = "fused"
    force_use_huggingface: bool = False
    activation_gpu_limit: float = 0.0
    basic_modules: Optional[list[str]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["veomni"], f"strategy {self.strategy} not supported"

        replacements = {
            "flash_attention_2": "veomni_flash_attention_2_with_sp",
            "flash_attention_3": "veomni_flash_attention_3_with_sp",
            "flash_attention_4": "veomni_flash_attention_4_with_sp",
        }
        if self.attn_implementation in replacements:
            new_impl = replacements[self.attn_implementation]
            logger.info(f"Replacing attn_implementation from '{self.attn_implementation}' to '{new_impl}'")
            self.attn_implementation = new_impl


@dataclass
class TorchtitanEngineConfig(EngineConfig):

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    reshard_after_forward: Literal["default", "always", "never"] = "default"
    forward_prefetch: bool = False
    use_orig_params: bool = False
    mixed_precision: bool = False
    offload_policy: bool = False
    use_torch_compile: bool = True
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    data_parallel_size: int = 1
    data_parallel_replicate_size: int = 1
    data_parallel_shard_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    attn_type: str = "flex"
    max_seq_len: Optional[int] = None
    strategy: str = "torchtitan"
    seed: int = 42
    full_determinism: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["torchtitan"], f"strategy {self.strategy} not supported"


@dataclass
class AutomodelEngineConfig(EngineConfig):
 

    strategy: str = "automodel"
    distributed_strategy: str = "fsdp2"
    # Parallelism sizes
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    dp_replicate_size: int = 1
    sequence_parallel: bool = False
    defer_fsdp_grad_sync: bool = True
    # Model settings
    activation_checkpointing: bool = False
    enable_fp8: bool = False
    enable_compile: bool = False
    model_dtype: str = "fp32"
    attn_implementation: str = "flash_attention_2"
    # Backend settings
    backend_config: dict = field(default_factory=dict)
    # MoE settings
    moe_config: dict = field(default_factory=dict)
    # Mixed precision policy
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_output_dtype: str = "bf16"
    # Entropy computation
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy == "automodel", f"strategy must be 'automodel', got {self.strategy}"
        assert self.distributed_strategy in ["fsdp2", "megatron_fsdp", "ddp"], (
            f"distributed_strategy {self.distributed_strategy} not supported"
        )
        assert self.pp_size == 1, "Pipeline parallelism (pp_size > 1) is not yet supported for automodel backend"


@dataclass
class MindSpeedEngineConfig(McoreEngineConfig):

    strategy: str = "mindspeed_llm"
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    mm_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert self.strategy in ["mindspeed_llm", "mindspeed_mm"], f"strategy {self.strategy} not supported"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class TrainingWorkerConfig(BaseConfig):
    model_type: str = None  # model type (language_model/value_model)
    model_config: HFModelConfig | DiffusionModelConfig = None
    engine_config: EngineConfig = None
    optimizer_config: OptimizerConfig = None
    checkpoint_config: CheckpointConfig = None
    profiler_config: ProfilerConfig = None
    # automatically select engine and optimizer function.
    # This function takes model config and the device name as parameter.
    # Users can pass in a higher-order function to take more parameters
    auto_select_engine_optim_fn: Callable[["HFModelConfig", str], tuple["EngineConfig", "OptimizerConfig"]] = None
