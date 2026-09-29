

from verl.utils.fp8_utils import FP8QuantizerHelper


class SGLangFP8QuantizerHelper(FP8QuantizerHelper):
    def __init__(self, quant_config):
        super().__init__(quant_config)
