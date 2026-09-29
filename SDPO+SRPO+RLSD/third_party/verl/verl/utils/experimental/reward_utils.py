

import base64
from io import BytesIO

from PIL import Image


def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_image_text}"
    return base64_image


def prepare_query_for_multi_modal(image_base64: str) -> list:
    query = [
        {
            "type": "image_url",
            "image_url": {"url": image_base64},
        },
    ]
    return query
