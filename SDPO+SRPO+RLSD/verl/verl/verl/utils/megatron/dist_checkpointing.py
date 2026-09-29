
import megatron.core
import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from packaging import version


def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
):
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    # Save model sharded state dicts
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata
    return dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
