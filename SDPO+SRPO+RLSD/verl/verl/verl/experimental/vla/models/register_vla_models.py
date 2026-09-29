
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

from verl.utils.transformers_compat import get_auto_model_for_vision2seq

from .openvla_oft.configuration_prismatic import OpenVLAConfig
from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from .openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from .pi0_torch import PI0ForActionPrediction, PI0TorchConfig

_REGISTERED_MODELS = {
    "openvla_oft": False,
    "pi0_torch": False,
}
AutoModelForVision2Seq = get_auto_model_for_vision2seq()


def register_openvla_oft() -> None:
    """Register the OpenVLA OFT model and processors."""
    if _REGISTERED_MODELS["openvla_oft"]:
        return

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    _REGISTERED_MODELS["openvla_oft"] = True


def register_pi0_torch_model() -> None:
    """Register the PI0 wrapper with the HF auto classes."""
    if _REGISTERED_MODELS["pi0_torch"]:
        return

    AutoConfig.register("pi0_torch", PI0TorchConfig)
    AutoModelForVision2Seq.register(PI0TorchConfig, PI0ForActionPrediction)

    _REGISTERED_MODELS["pi0_torch"] = True


def register_vla_models() -> None:
    """Register all custom VLA models with Hugging Face."""
    register_openvla_oft()
    register_pi0_torch_model()
