
from abc import ABC, abstractmethod

import torch


class Pi0Input(ABC):
    def __init__(self):
        # three images for pi0 input with keys:
        # [
        #     'observation.images.cam_high',
        #     'observation.images.cam_left_wrist',
        #     'observation.images.cam_right_wrist',
        # ],
        # each with shape (B, C, H, W)
        self.images: dict[str, torch.Tensor] = {}

        # image masks corresponding to the images, each with shape (B,)
        self.img_masks: list[torch.Tensor] = []

        # task description as a list of strings
        self.task: list[str] = []

        # robot state with shape (B, state_dim)
        self.state: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_env_obs(cls, env_obs) -> "Pi0Input": ...


class Pi0Output:
    def __init__(self):
        self.action: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_model_output(cls, model_output) -> "Pi0Output": ...
