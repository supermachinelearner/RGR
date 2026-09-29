

import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights):
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            async for name, weight in ensure_async_iterator(weights):
                if offset + weight.nbytes > self.bucket_size:
                    get_torch_device().synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0
                assert offset + weight.nbytes <= self.bucket_size, (
                    f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                    f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                )
                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes
            get_torch_device().synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()


class BucketedWeightReceiver:

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: callable):
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
