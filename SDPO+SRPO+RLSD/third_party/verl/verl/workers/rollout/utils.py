
import asyncio
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


class _UvicornServerAutoPort(uvicorn.Server):

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    if (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        prompt_ids = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids), dtype=bool)
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)
        mask[1:] &= ~(is_value[1:] & is_value[:-1])
        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
