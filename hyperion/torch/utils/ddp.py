"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .devices import open_device


def add_ddp_args(parser: ArgumentParser) -> None:
    """Adds common distributed-training CLI arguments to a parser."""
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="number of gpus, if 0 it uses cpu (deprecated)",
    )
    # parser.add_argument(
    #     "--node-id", type=int, default=0, help="node id for distributed training"
    # )
    # parser.add_argument(
    #     "--num-nodes",
    #     type=int,
    #     default=1,
    #     help="number of nodes in which we distribute the training",
    # )
    # parser.add_argument(
    #     "--master-addr", default="127.0.0.1", help="address of the master node"
    # )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="optional override for MASTER_PORT; if None, use launcher environment",
    )


def filter_ddp_args(**kwargs: Any) -> Dict[str, Any]:
    """Returns only keyword arguments relevant to DDP initialization."""
    valid_args = ("master_port",)
    args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
    return args


def ddp_init(master_port: Optional[int] = None) -> Tuple[torch.device, int, int]:
    """Initializes torch distributed process group and returns local process info.

    This function expects torchrun-style environment variables
    (``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``).
    For single-process execution (``world_size == 1``), it only opens a device.

    Args:
        master_port: Optional override for ``MASTER_PORT``.

    Returns:
        Tuple of ``(device, rank, world_size)``.
    """
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device = open_device(1)
        return device, 0, 1

    required_vars = ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        raise RuntimeError(
            f"Missing required distributed env vars: {missing_vars}. "
            "Launch with torchrun or provide MASTER_PORT override."
        )

    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2) Set the correct local device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info(
        f"init-process-group rank={rank} world_size={world_size} master={master_addr}:{master_port} gpu_id={local_rank}"
    )
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=torch.distributed.constants.default_pg_timeout,  # optional; keep default or tune
        device_id=device,  # tells NCCL which GPU this rank owns; avoids barrier device-guess warning
    )
    logging.info(
        f"done init-process-group rank={rank} world_size={world_size} master={master_addr}:{master_port} gpu_id={local_rank}"
    )
    # 4) Sanity touch (on this device)

    torch.empty(0, device=device)

    return device, rank, world_size


# def legacy_ddp_init(
#     gpu_id: int,
#     num_gpus: int,
#     node_id: int = 0,
#     num_nodes: int = 1,
#     master_addr: str = "localhost",
#     master_port: Optional[int] = None,
# ) -> Tuple[torch.device, int, int]:
#     """Legacy DDP initialization path using explicit rank/world-size arguments.

#     Args:
#         gpu_id: Local GPU id for this process.
#         num_gpus: Number of GPUs per node.
#         node_id: Node index in multi-node runs.
#         num_nodes: Number of nodes.
#         master_addr: Master node address.
#         master_port: Master node port.

#     Returns:
#         Tuple of ``(device, rank, world_size)``.
#     """
#     rank = node_id * num_gpus + gpu_id
#     world_size = num_nodes * num_gpus

#     if world_size == 1:
#         device = open_device(num_gpus)
#         return device, 0, 1

#     os.environ["MASTER_ADDR"] = str(master_addr)
#     os.environ["MASTER_PORT"] = str(master_port)

#     logging.info(
#         f"init ddp rank={rank} world_size={world_size} master={master_addr}:{master_port} gpu_id={gpu_id}"
#     )
#     dist.init_process_group(
#         "nccl",
#         rank=rank,
#         world_size=world_size,
#     )
#     torch.cuda.set_device(rank)
#     torch.tensor([0]).to(gpu_id)
#     device = torch.device("cuda", gpu_id)
#     return device, rank, world_size


def ddp_cleanup() -> None:
    """Destroys the active process group if it exists."""
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def ddp_wait_for_all_procs() -> None:
    """Synchronizes all processes with a barrier when DDP is initialized."""
    if dist.is_initialized():
        dist.barrier()


def ddp_get_rank_world_size() -> Tuple[int, int]:
    """Returns current ``(rank, world_size)`` or ``(0, 1)`` when not initialized."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def ddp_get_rank() -> int:
    """Returns current process rank or ``0`` when DDP is not initialized."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


class TorchDDP(nn.parallel.DistributedDataParallel):
    """DDP wrapper that forwards missing attributes to the wrapped module."""

    def __getattr__(self, name: str) -> Any:
        """Resolves attributes from this wrapper first, then wrapped ``module``."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
