
import warnings
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = [
    "OptimizerConfig",
    "FSDPOptimizerConfig",
    "McoreOptimizerConfig",
    "build_optimizer",
    "VeOmniOptimizerConfig",
    "TorchtitanOptimizerConfig",
    "AutomodelOptimizerConfig",
]


@dataclass
class OptimizerConfig(BaseConfig):


    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None
    zero_indexed_step: bool = True

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):


    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


@dataclass
class TorchtitanOptimizerConfig(OptimizerConfig):

    name: str = "AdamW"
    eps: float = 1e-8
    decay_type: str = "linear"
    min_lr_factor: float = 0.0


@dataclass
class AutomodelOptimizerConfig(OptimizerConfig):


    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    init_lr_ratio: Optional[float] = 0.1
    min_lr_ratio: Optional[float] = 0.01
    lr_scheduler_type: str = "cosine"
    wd_incr_style: str = "constant"
    num_cycles: float = 0.5
    zero_indexed_step: bool = True
    # Common optimizer kwargs
    eps: float = 1e-8
    master_weights: bool = False
    store_param_remainders: bool = False
    exp_avg_dtype: Optional[str] = None  # "fp32", "bf16", "fp16", or "torch.float32" etc.
    exp_avg_sq_dtype: Optional[str] = None  # "fp32", "bf16", "fp16", or "torch.float32" etc.
    master_weight_dtype: Optional[str] = None  # "fp32", "bf16", "fp16", or "torch.float32" etc.
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        assert self.lr_scheduler_type in ["constant", "cosine", "linear", "inverse-square-root"]
        return super().__post_init__()


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    import importlib

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = config.optimizer.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)
