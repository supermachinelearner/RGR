
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from verl.protocol import DataProto

RawRewardFn = Callable[..., Any]


class AbstractRewardManager(ABC):
    @abstractmethod
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        pass

    def _extract_reward_from_rm_scores(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        if "rm_scores" not in data.batch.keys():
            return None
        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        else:
            return data.batch["rm_scores"]