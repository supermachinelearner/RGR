

import logging
from typing import Any

import torch

from verl import DataProto
from verl.experimental.vla.naive_rollout_rob import NaiveRolloutRob
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)

__all__ = ["PI0RolloutRob"]


class PI0RolloutRob(NaiveRolloutRob):
    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module,
        tokenizer: Any,
    ):
        self.model_config = model_config
        self.module = module
        self.tokenizer = tokenizer

        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self.module, "sample_actions")
        register_fsdp_forward_method(self.module, "sac_forward_state_features")
        register_fsdp_forward_method(self.module, "sac_forward_critic")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            prompts.to(get_device_id())
            validate = bool(prompts.meta_info.get("validate", False))
            output, s, a = self.module.sample_actions(
                prompts,
                tokenizer=self.tokenizer,
                validate=validate,
            )
            state_features = self.module.sac_forward_state_features(s)
            critic_value = (
                self.module.sac_forward_critic(
                    {"full_action": a["full_action"]},
                    state_features,
                    use_target_network=False,
                    method="min",
                    requires_grad=False,
                )
                .detach()
                .float()
                .reshape(-1)
            )

        tensor_batch = {
            "action": output.action,
            "full_action": a["full_action"],
            "images": s["images"],
            "image_masks": s["image_masks"],
            "lang_tokens": s["lang_tokens"],
            "lang_masks": s["lang_masks"],
            "states": s["states"],
            "critic_value": critic_value,
        }

        ret = DataProto.from_dict(tensor_batch)

        return ret
