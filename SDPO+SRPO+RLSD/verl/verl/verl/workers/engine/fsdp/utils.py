
import logging
import os

import torch
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name, is_npu_available

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_npu_fsdp_patches():
    if is_npu_available:
        try:
            import verl.models.transformers.npu_patch  # noqa

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.info("Applied NPU patches for FSDP backend")
        except Exception as e:
            logger.warning(f"Failed to apply NPU patches: {e}")


def create_device_mesh(world_size, fsdp_size):
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy
