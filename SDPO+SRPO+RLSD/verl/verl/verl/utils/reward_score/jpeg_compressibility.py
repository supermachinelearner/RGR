
import io

import numpy as np
import torch
from PIL import Image


def jpeg_incompressibility():
    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers, strict=False):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts):
        rew, meta = jpeg_fn(images, prompts)
        return -rew / 500, meta

    return _fn


def compute_score(solution_image):
    """The scoring function for JPEG compressibility.

    Args:
        solution_image: the solution image or video, in shape (C, H, W) or (N, C, H, W).
    """
    if isinstance(solution_image, torch.Tensor) and solution_image.ndim == 3:
        solution_image = solution_image.unsqueeze(0)
    score = jpeg_compressibility()(solution_image, None)[0]
    return score
