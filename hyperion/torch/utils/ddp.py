"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import datetime
import logging
import os

from fairscale.nn.data_parallel import \
    FullyShardedDataParallel as FullyShardedDDP
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

import torch
import torch.distributed as dist
import torch.nn as nn

from .devices import open_device


def add_ddp_args(parser):

    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus, if 0 it uses cpu"
    )
    parser.add_argument(
        "--node-id", type=int, default=0, help="node id for distributed training"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes in which we distribute the training",
    )
    parser.add_argument(
        "--master-addr", default="localhost", help="address of the master node"
    )
    parser.add_argument(
        "--master-port",
        default="1234",
        help="port of the master node, if None it will be random",
    )


def filter_ddp_args(**kwargs):
    valid_args = ("num_gpus", "node_id", "num_nodes", "master_addr", "master_port")
    args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
    return args


def ddp_init(
    gpu_id, num_gpus, node_id=0, num_nodes=1, master_addr="localhost", master_port=None
):

    rank = node_id * num_gpus + gpu_id
    world_size = num_nodes * num_gpus

    if world_size == 1:
        device = open_device(num_gpus)
        return device, 0, 1

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    logging.info(
        f"init ddp rank={rank} world_size={world_size} master={master_addr}:{master_port} gpu_id={gpu_id}" 
    )
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.tensor([0]).to(gpu_id)
    device = torch.device('cuda', gpu_id)
    return device, rank, world_size
    # return gpu_id, rank, world_size


def ddp_cleanup():
    try:
        dist.destroy_process_group()
    except:
        pass


def ddp_wait_for_all_procs():
    if dist.is_initialized():
        dist.barrier()


def ddp_get_rank_world_size():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def ddp_get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


class TorchDDP(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class FairShardedDDP(ShardedDDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class FairFullyShardedDDP(FullyShardedDDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
