
import modelopt.torch.quantization as mtq
import torch.nn as nn
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> dict:
    cfg = {}
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg[key] = {"enable": False}
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    if qat_mode != "w4a16":
        raise ValueError(f"Only 'w4a16' is supported, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = {
        **_NVFP4_W4A16_QUANTIZER_CFG,
        **_default_disabled_quantizer_cfg,
        **ignore_cfg,
    }
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
