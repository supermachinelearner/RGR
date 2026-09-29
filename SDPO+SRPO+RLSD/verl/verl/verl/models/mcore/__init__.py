
from verl.models.mcore.patch import apply_patch_megatron_v012_with_torch_v28

from .registry import (
    get_mcore_engine_forward_fn,
    get_mcore_forward_fn,
    get_mcore_forward_fused_fn,
    get_mcore_forward_fused_model_engine_fn,
    get_mcore_weight_converter,
    hf_to_mcore_config,
    init_mcore_model,
)

__all__ = [
    "hf_to_mcore_config",
    "init_mcore_model",
    "get_mcore_forward_fn",
    "get_mcore_weight_converter",
    "get_mcore_forward_fused_fn",
    "get_mcore_engine_forward_fn",
    "get_mcore_forward_fused_model_engine_fn",
]

apply_patch_megatron_v012_with_torch_v28()
