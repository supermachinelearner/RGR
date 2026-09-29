

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import BaseModelConfig, CheckpointConfig
from verl.utils.profiler import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig, MindSpeedEngineConfig, TorchtitanEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "CriticConfig",
    "FSDPCriticConfig",
    "McoreCriticConfig",
    "TorchTitanCriticConfig",
    "FSDPCriticModelCfg",
    "MindSpeedCriticConfig",
]


@dataclass
class CriticConfig(BaseConfig):


    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_micro_batch_size_per_gpu",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    enable: Optional[bool] = None
    rollout_n: int = 1
    ppo_mini_batch_size: int = 1
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 32768
    # deprecate this
    forward_max_token_len_per_gpu: int = 32768
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_max_token_len_per_gpu: int = 32768
    ppo_epochs: int = 1
    data_loader_seed: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    ppo_micro_batch_size: Optional[int] = None
    engine: BaseConfig = field(default_factory=BaseConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: HFModelConfig = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        assert self.strategy != MISSING

        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.ppo_micro_batch_size, self.ppo_micro_batch_size_per_gpu, "critic")

            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"[critic] ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )

    def validate(self, n_gpus: int, train_batch_size: int):

        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"critic.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
 
        param = "micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreCriticConfig(CriticConfig):

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    load_weight: bool = True

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate Megatron critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)

    def __post_init__(self):
        """Validate Megatron critic configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPCriticConfig(CriticConfig):


    _mutable_fields = CriticConfig._mutable_fields | {
        "forward_micro_batch_size",
        "forward_micro_batch_size_per_gpu",
    }

    strategy: str = "fsdp"
    fsdp: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.fsdp
        # Sync strategy to engine config so engine_workers can pick the right FSDP version.
        # EngineConfig.strategy defaults to None, so without this, engine_workers.py always
        # falls back to FSDP1 even when critic.strategy="fsdp2".
        object.__setattr__(self.engine, "strategy", self.strategy)

        if self.strategy in {"fsdp", "fsdp2"}:
            if self.ulysses_sequence_parallel_size > 1:
                if not self.model.get("use_remove_padding", False):
                    raise ValueError(
                        "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate FSDP critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)

        if not self.use_dynamic_bsz:
            sp_size = self.ulysses_sequence_parallel_size
            if self.ppo_micro_batch_size is not None:
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"critic.ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )


@dataclass
class TorchTitanCriticConfig(CriticConfig):


    strategy: str = "torchtitan"
    torchtitan: TorchtitanEngineConfig = field(default_factory=TorchtitanEngineConfig)

    def __post_init__(self):
        """Validate TorchTitan critic configuration parameters."""
        super().__post_init__()
        self.engine = self.torchtitan


@dataclass
class FSDPCriticModelCfg(BaseModelConfig):


    use_shm: bool = False
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = True
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str | list[str] = "all-linear"
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})


@dataclass
class MindSpeedCriticConfig(CriticConfig):

    strategy: str = "mindspeed"
    nccl_timeout: int = 600
    mindspeed: MindSpeedEngineConfig = field(default_factory=MindSpeedEngineConfig)
    load_weight: bool = True

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate mindspeed critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)
