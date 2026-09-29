

from typing import Callable

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER: dict[str, type[RewardManagerBase]] = {}


def register(name: str) -> Callable[[type[RewardManagerBase]], type[RewardManagerBase]]:
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls: type[RewardManagerBase]) -> type[RewardManagerBase]:
        if name in REWARD_MANAGER and REWARD_MANAGER[name] != cls:
            raise ValueError(f"reward manager {name} has already been registered: {REWARD_MANAGER[name]} vs {cls}")
        REWARD_MANAGER[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER[name]
