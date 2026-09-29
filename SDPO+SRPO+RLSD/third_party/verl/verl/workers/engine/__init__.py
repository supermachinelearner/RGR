
from .base import BaseEngine, EngineRegistry
from .fsdp import DiffusersFSDPEngine, FSDPEngine, FSDPEngineWithLMHead

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "FSDPEngine",
    "FSDPEngineWithLMHead",
]

if DiffusersFSDPEngine is not None:
    __all__.append("DiffusersFSDPEngine")

try:
    from .torchtitan import TorchTitanEngine, TorchTitanEngineWithLMHead

    __all__ += ["TorchTitanEngine", "TorchTitanEngineWithLMHead"]
except ImportError:
    TorchTitanEngine = None
    TorchTitanEngineWithLMHead = None

try:
    from .veomni import VeOmniEngine, VeOmniEngineWithLMHead

    __all__ += ["VeOmniEngine", "VeOmniEngineWithLMHead"]
except ImportError:
    VeOmniEngine = None
    VeOmniEngineWithLMHead = None

try:
    from .automodel import AutomodelEngine, AutomodelEngineWithLMHead

    __all__ += ["AutomodelEngine", "AutomodelEngineWithLMHead"]
except ImportError:
    AutomodelEngine = None
    AutomodelEngineWithLMHead = None

try:
    from .mindspeed import MindspeedEngineWithLMHead, MindSpeedLLMEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead", "MindSpeedLLMEngineWithLMHead"]
except ImportError:
    MindspeedEngineWithLMHead = None
    MindSpeedLLMEngineWithLMHead = None

try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead"]
except ImportError:
    MegatronEngine = None
    MegatronEngineWithLMHead = None
