
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):


    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):

    _mutable_fields = BaseConfig._mutable_fields

    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class DistillationConfig(BaseConfig):


    _mutable_fields = BaseConfig._mutable_fields

    enabled: bool = False
    num_workers: int = 8
    teacher_model: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.teacher_model.inference.max_model_len
        student_prompt_length = self.teacher_model.inference.prompt_length
        student_response_length = self.teacher_model.inference.response_length
        if self.enabled:
            required_context_len = student_prompt_length + student_response_length + 1
            if max_model_len is not None and required_context_len > max_model_len:
                raise ValueError(
                    "Distillation teacher inference requires room for the student prompt, the full student "
                    f"response, and one generated token, but got {student_prompt_length=}, "
                    f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
                )

        self.teacher_model.inference.prompt_length = (
            self.teacher_model.inference.prompt_length + self.teacher_model.inference.response_length
        )
        self.teacher_model.inference.response_length = 1
        engine_name = self.teacher_model.inference.name
        engine_kwargs = self.teacher_model.inference.engine_kwargs
        if not self.distillation_loss.loss_settings.use_topk or self.distillation_loss.topk is None or not self.enabled:
            return
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = self.distillation_loss.topk
                    max_logprobs = self.distillation_loss.topk
                if max_logprobs < self.distillation_loss.topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({self.distillation_loss.topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )
