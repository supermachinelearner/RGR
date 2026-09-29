from typing import Callable

from verl.workers.reward_manager.abstract import AbstractRewardManager

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER_REGISTRY: dict[str, type[AbstractRewardManager]] = {}


def register(name: str) -> Callable[[type[AbstractRewardManager]], type[AbstractRewardManager]]:

    def decorator(cls: type[AbstractRewardManager]) -> type[AbstractRewardManager]:
        if name in REWARD_MANAGER_REGISTRY and REWARD_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"Reward manager {name} has already been registered: {REWARD_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[AbstractRewardManager]:
    if name not in REWARD_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]
