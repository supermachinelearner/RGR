

from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl.utils.device import get_device_name
from verl.workers.config import DiffusionModelConfig

from .base import DiffusionModelBase


def prepare_model_inputs(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_embeds_mask: torch.Tensor,
    micro_batch: TensorDict,
    step: int,
) -> tuple[dict, dict]:

    return DiffusionModelBase.get_class(model_config).prepare_model_inputs(
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    )


def build_scheduler(model_config: DiffusionModelConfig) -> SchedulerMixin:
    return DiffusionModelBase.get_class(model_config).build_scheduler(model_config)


def set_timesteps(scheduler: SchedulerMixin, model_config: DiffusionModelConfig):
    DiffusionModelBase.get_class(model_config).set_timesteps(scheduler, model_config, get_device_name())


def forward_and_sample_previous_step(
    module: ModelMixin,
    scheduler: SchedulerMixin,
    model_config: DiffusionModelConfig,
    model_inputs: dict,
    negative_model_inputs: Optional[dict],
    scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
    step: int,
):
    return DiffusionModelBase.get_class(model_config).forward_and_sample_previous_step(
        module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
    )
