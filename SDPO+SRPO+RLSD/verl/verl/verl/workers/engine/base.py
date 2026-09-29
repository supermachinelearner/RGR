

from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_name
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids


class BaseEngine:


    def initialize(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        """Whether optimizer offloading is enabled."""
        raise NotImplementedError

    def train_mode(self, **kwargs):
        raise NotImplementedError

    def eval_mode(self, **kwargs):
        raise NotImplementedError

    def optimizer_zero_grad(self):

        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform an optimization step using the optimizer.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        raise NotImplementedError

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        maybe_fix_3d_position_ids(data)

        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        if self.is_mp_src_rank_with_outputs():
            assert "grad_norm" not in outputs["metrics"]
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> Any:
        maybe_fix_3d_position_ids(data)

        with torch.no_grad():
            outputs = self.forward_backward_batch(data, loss_function, forward_only=True)
        return outputs

    def get_per_tensor_param(self) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        raise NotImplementedError

    def get_data_parallel_size(self):
        raise NotImplementedError

    def get_data_parallel_rank(self):
        raise NotImplementedError

    def get_data_parallel_group(self):
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        if not model:
            assert not optimizer and not grad, "Model must be moved to device along with optimizer and grad"

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        raise NotImplementedError

    def disable_adapter(self) -> ContextManager:
        return nullcontext()


class BaseEngineCtx:
    def __init__(self, engine: BaseEngine, mode, **kwargs):

        self.engine = engine
        self.mode = mode
        assert self.mode in ("train", "eval")
        self.disable_auto_offload = kwargs.pop("disable_auto_offload", False)

    def _context_switch(self, device):
        if self.disable_auto_offload:
            return
        if self.mode == "eval":
            self.engine.to(device=device, model=self.engine.is_param_offload_enabled, optimizer=False, grad=False)
        elif self.mode == "train":
            self.engine.to(
                device=device,
                model=self.engine.is_param_offload_enabled,
                optimizer=self.engine.is_optimizer_offload_enabled,
                grad=self.engine.is_param_offload_enabled,
            )

    def __enter__(self):
        self._context_switch(get_device_name())
        self.engine.mode = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu")
        self.engine.mode = None


class EngineRegistry:

    _engines = {}

    @classmethod
    def register(cls, model_type: str, backend: list[str] | str, device: list[str] | str = "cuda"):


        def decorator(engine_class):
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            for current_backend in backends:
                for current_device in devices:
                    if current_backend not in cls._engines[model_type]:
                        cls._engines[model_type][current_backend] = {}
                    if current_device not in cls._engines[model_type][current_backend]:
                        cls._engines[model_type][current_backend][current_device] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = get_device_name()
        assert device in cls._engines[model_type][backend], (
            f"Unknown device: {device} for model_type: {model_type} and backend: {backend}"
        )
        return cls._engines[model_type][backend][device]

    @classmethod
    def new(cls, model_type, backend, *args, **kwargs):
        engine_cls = cls.get_engine_cls(model_type, backend)
        return engine_cls(*args, **kwargs)
