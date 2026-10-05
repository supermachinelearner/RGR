
from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "register",
    "get_reward_manager_cls",
]

# Import experimental reward managers to ensure they are registered
try:
    from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager  # noqa: F401

    __all__.append("RateLimitedRewardManager")
except ImportError:
    pass  # Optional dependency, may not be available
