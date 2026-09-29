
import importlib.metadata
from functools import lru_cache
from typing import Optional

from packaging import version

try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


@lru_cache
def get_auto_model_for_vision2seq():
    """Return the available VL auto model class across transformers versions."""

    try:
        # Prefer the newer class when available. In transformers 4.x this class has
        # a broader mapping than AutoModelForVision2Seq, and AutoModelForVision2Seq
        # is deprecated for removal in v5.
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq

    return AutoModelForImageTextToText


def unpack_visual_output(visual_output):
    """Unpack the output from the visual encoder, handling both tuple and object return types.

    Newer versions of transformers return an object with `pooler_output` and `deepstack_features`
    attributes instead of a plain tuple.
    """
    if hasattr(visual_output, "pooler_output"):
        # For newer versions(>=5.0.0) of transformers, return the pooler_output and deepstack_features
        if hasattr(visual_output, "deepstack_features"):
            return visual_output.pooler_output, visual_output.deepstack_features
        else:
            return visual_output.pooler_output, None
    if isinstance(visual_output, tuple):
        return visual_output
    else:
        return visual_output, None
