

from abc import ABC, abstractmethod
from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl.workers.config import DiffusionModelConfig


class DiffusionModelBase(ABC):

    _registry: dict[str, type["DiffusionModelBase"]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers a subclass under *name*."""

        def decorator(subclass: type["DiffusionModelBase"]) -> type["DiffusionModelBase"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_config: DiffusionModelConfig) -> type["DiffusionModelBase"]:
        """Return the registered subclass for ``model_config.architecture``."""
        if model_config.architecture not in cls._registry and model_config.external_lib is not None:
            from verl.utils.import_utils import import_external_libs

            import_external_libs(model_config.external_lib)

        try:
            return cls._registry[model_config.architecture]
        except KeyError:
            registered = list(cls._registry)
            raise NotImplementedError(
                f"No diffusion model registered for architecture={model_config.architecture!r}. "
                f"Registered: {registered}. "
                f"Set ``external_lib`` in DiffusionModelConfig to load your implementation."
            ) from None

    @classmethod
    @abstractmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> SchedulerMixin:
        """Build and configure the diffusion scheduler for this model.
        The returned scheduler should have timesteps and sigmas already set.

        Args:
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
        """
        pass

    @classmethod
    @abstractmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """Set timesteps and sigmas on the scheduler and move them to *device*.

        Args:
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            device (str): the device to move the timesteps and sigmas to.
        """
        pass

    @classmethod
    @abstractmethod
    def prepare_model_inputs(
        cls,
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

        pass

    @classmethod
    @abstractmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):

        pass
