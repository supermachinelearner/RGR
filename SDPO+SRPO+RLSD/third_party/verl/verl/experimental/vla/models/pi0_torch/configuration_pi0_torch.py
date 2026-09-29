

from transformers import PretrainedConfig


class PI0TorchConfig(PretrainedConfig):
    model_type = "pi0_torch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_norm_stats = kwargs.get("state_norm_stats", {})
        self.action_norm_stats = kwargs.get("action_norm_stats", {})
        self.pi05_enabled = kwargs.get("pi05_enabled", False)
