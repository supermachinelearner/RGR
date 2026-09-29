
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig, RolloutCorrectionConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.utils.qat import QATConfig

from .engine import (
    FSDPEngineConfig,
    McoreEngineConfig,
    MindSpeedEngineConfig,
    TorchtitanEngineConfig,
    VeOmniEngineConfig)
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
    "VeOmniActorConfig",
    "QATConfig",
    "TorchTitanActorConfig",
    "MindSpeedActorConfig",
]


@dataclass
class RouterReplayConfig(BaseConfig):

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):

        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    # Self-distillation params (OPSD / SDPO / RLSD)
    beta: float = 0.0
    jsd_token_clip: float = 0.05
    teacher_mode: str = "fixed"
    rationale_source: str = "self"
    rationale_model: str = "qwen3.5-plus"
    rationale_temperature: float = 1.0
    rationale_max_tokens: int = 8192
    rationale_api_timeout_seconds: float = 120.0
    rationale_api_max_retries: int = 3
    rationale_api_retry_backoff_seconds: float = 2.0
    rationale_api_max_concurrency: int = 8
    use_rationale_judge: bool = False
    rationale_judge_model: str = "qwen3.5-plus"
    rationale_judge_temperature: float = 0.0
    rationale_judge_max_tokens: int = 32
    rationale_judge_timeout_seconds: float = 60.0
    rationale_judge_max_retries: int = 3
    rationale_judge_retry_backoff_seconds: float = 2.0
    rationale_judge_max_concurrency: int = 8
    full_logit_distill: bool = True
    distill_add_tail: bool = True
    alpha: float = 0.5
    top_k_distill: int = 0
    is_clip: float = 2.0
    ema_decay: float = 0.95
    epsilon: float = 0.2
    epsilon_w: float = 0.2
    tau: Optional[float] = None
    lam: float = 0.5
    lam_decay_steps: int = 50
    teacher_sync_interval: int = 10
    rlcsd2_w_tau: Optional[float] = None
    rlcsd2_snr_tau: Optional[float] = None
    rlcsd2_snr_alpha: Optional[float] = None
    rlcsd2_snr_eps: Optional[float] = None
    rlcsd3_w_tau: Optional[float] = None
    rlcsd3_snr_tau: Optional[float] = None
    rlcsd3_snr_alpha: Optional[float] = None
    rlcsd3_snr_eps: Optional[float] = None
    rlcsd_tau: Optional[float] = None
    rlcsd_beta: Optional[float] = None
    rlcsd_lam: Optional[float] = None
    rlcsd_delta: Optional[float] = None
    rlcsd_eta: Optional[float] = None
    rlcsd_residual_clip_low: Optional[float] = None
    rlcsd_residual_clip_high: Optional[float] = None
    rlcsd_k_max: Optional[int] = None
    srpo_beta: Optional[float] = None
    # Custom hidden-state penalty
    use_hidden_penalty: bool = False
    hidden_penalty_weight: float = 0.0


    hidden_snapshot_interval: int = 25
    hidden_snapshot_last_step: int = 175

@dataclass
class ActorConfig(BaseConfig):


    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config"}

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)
    qat: QATConfig = field(default_factory=QATConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(

                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.fsdp_config
        object.__setattr__(self.engine, "strategy", self.strategy)
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )


@dataclass
class VeOmniActorConfig(ActorConfig):

    strategy: str = "veomni"
    veomni: VeOmniEngineConfig = field(default_factory=VeOmniEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.veomni


@dataclass
class TorchTitanActorConfig(ActorConfig):
    strategy: str = "torchtitan"
    torchtitan: TorchtitanEngineConfig = field(default_factory=TorchtitanEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.torchtitan


@dataclass
class MindSpeedActorConfig(ActorConfig):


    strategy: str = "mindspeed"
    load_weight: bool = True
    mindspeed: MindSpeedEngineConfig = field(default_factory=MindSpeedEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.mindspeed
