
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.trainer.config.config import ModuleConfig

from .rollout import RolloutConfig

__all__ = ["SandboxFusionConfig", "RewardConfig", "RewardModelConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class RewardManagerConfig(BaseConfig):

    source: str = "register"
    name: str = "naive"
    module: Optional[ModuleConfig] = field(default_factory=ModuleConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.source == "register":
            from verl.experimental.reward_loop.reward_manager.registry import REWARD_MANAGER

            assert self.name in REWARD_MANAGER, (
                f"Reward manager is not registered: {self.name=} ,{REWARD_MANAGER.keys()=}"
            )
        elif self.source == "importlib":
            assert self.module is not None and self.module.path is not None, (
                "When source is importlib, module.path should be set."
            )


@dataclass
class SandboxFusionConfig(BaseConfig):
    url: Optional[str] = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    enable: bool = False
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class RewardConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    # reward manager args
    num_workers: int = 8
    reward_manager: RewardManagerConfig = field(default_factory=RewardManagerConfig)

    # reward model args
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    # sandbox fusion args
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
