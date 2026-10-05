
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

__all__ = ["HFModelConfig", "DiffusionModelConfig", "MtpConfig"]


@dataclass
class MtpConfig(BaseConfig):


    enable: bool = False
    enable_train: bool = False
    enable_rollout: bool = False

    detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1

    speculative_algorithm: str = "EAGLE"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4

    method: str = "mtp"
    num_speculative_tokens: int = 1


@dataclass
class HFModelConfig(BaseConfig):
    _mutable_fields = {
        "model_type",
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
        "mtp",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    model_type: str = "language_model"
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None
    use_shm: bool = False
    trust_remote_code: bool = False
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = True
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj","k_proj"]
    target_parameters: Optional[list[str]] = None  # for lora adapter on nn.Parameter

    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # TiledMLP configuration for memory-efficient MLP computation
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})

    architectures: Optional[list[str]] = None

    mtp: MtpConfig = field(default_factory=MtpConfig)

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        if (
            self.processor is not None
            and not getattr(self.processor, "chat_template", None)
            and getattr(self.tokenizer, "chat_template", None)
        ):
            self.processor.chat_template = self.tokenizer.chat_template

        if self.custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = self.custom_chat_template
            else:
                self.tokenizer.chat_template = self.custom_chat_template

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

        if self.target_modules is not None:
            if not isinstance(self.target_modules, (str | list)):
                raise TypeError(
                    "target_modules must be a string or a list of strings, "
                    f"but got {type(self.target_modules).__name__}"
                )
            if isinstance(self.target_modules, list):
                for x in self.target_modules:
                    if not isinstance(x, str):
                        raise TypeError(
                            f"All elements in target_modules list must be strings, but found {type(x).__name__}"
                        )

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer


@dataclass
class DiffusionModelConfig(BaseConfig):
    _mutable_fields = {
        "model_type",
        "tokenizer_path",
        "tokenizer",
        "processor",
        "local_path",
        "local_tokenizer_path",
        "architecture",
    }

    path: str = MISSING

    architecture: Optional[str] = None
    local_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None
    model_type: str = "diffusion_model"


    load_tokenizer: bool = True

    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    enable_gradient_checkpointing: bool = True

    # lora related.
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_init_weights: str = "gaussian"
    target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj","k_proj"]
    target_parameters: Optional[list[str]] = None  # for lora adapter on nn.Parameter
    exclude_modules: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    lora: dict[str, Any] = field(default_factory=dict)
    height: int = 512
    width: int = 512
    num_inference_steps: int = 10
    extra_configs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import_external_libs(self.external_lib)
        if self.tokenizer_path is None:
            self.tokenizer_path = os.path.join(self.path, "tokenizer")
        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        if self.architecture is None:
            import json

            model_index_path = os.path.join(self.local_path, "model_index.json")
            with open(model_index_path) as f:
                self.architecture = json.load(f)["_class_name"]

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(
                self.local_tokenizer_path, trust_remote_code=self.trust_remote_code, use_fast=True
            )
            if os.path.exists(os.path.join(self.local_path, "processor")):
                self.processor = hf_processor(
                    os.path.join(self.local_path, "processor"), trust_remote_code=self.trust_remote_code
                )
            else:
                self.processor = None

        if self.target_modules is not None:
            if not isinstance(self.target_modules, (str | list)):
                raise TypeError(
                    f"but got {type(self.target_modules).__name__}"
                )
            if isinstance(self.target_modules, list):
                for x in self.target_modules:
                    if not isinstance(x, str):
                        raise TypeError(
                            f"All elements in target_modules list must be strings, but found {type(x).__name__}"
                        )

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
