"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

TensorTree = Union[
    torch.Tensor,
    Dict[Any, torch.Tensor],
    List[torch.Tensor],
    Tuple[torch.Tensor, ...],
]
NumpyTree = Union[
    np.ndarray,
    Dict[Any, np.ndarray],
    List[np.ndarray],
    Tuple[np.ndarray, ...],
]
GpuIds = Optional[List[int]]


def open_device(
    num_gpus: int = 1, gpu_ids: GpuIds = None, find_free_gpu: bool = False
) -> torch.device:
    """Initializes and returns the primary compute device.

    Args:
        num_gpus: Number of GPUs requested. If 0, returns a CPU device.
        gpu_ids: Optional list of CUDA-visible GPU ids.
        find_free_gpu: If True, query free GPUs and set ``CUDA_VISIBLE_DEVICES``.

    Returns:
        The selected torch device.
    """
    if find_free_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu_ids is None:
            gpu_ids = find_free_gpus(num_gpus)

    if gpu_ids is not None and num_gpus > 0:
        if len(gpu_ids) == 0:
            raise RuntimeError(f"No free GPUs found for num_gpus={num_gpus}")
        if len(gpu_ids) < num_gpus:
            raise RuntimeError(
                f"Requested num_gpus={num_gpus}, but only found gpu_ids={gpu_ids}"
            )
        gpu_ids_str = ",".join([str(i) for i in gpu_ids])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    if num_gpus > 0 and torch.cuda.is_available():
        logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
        logging.info("init gpu device")
        device = torch.device("cuda", 0)
        torch.tensor([0]).to(device)
        # reserve the rest of gpus
        for n in range(1, num_gpus):
            device_n = torch.device("cuda", n)
            # torch.tensor([0]).to(device_n)
    else:
        logging.info("init cpu device")
        device = torch.device("cpu")

    return device


def find_free_gpus(num_gpus: int) -> List[int]:
    """Returns free GPU ids from the ``free-gpu`` helper."""
    try:
        result = subprocess.run(
            ["free-gpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        raw_ids = result.stdout.strip()
        gpu_ids = [int(i) for i in re.split(r"[,\s]+", raw_ids) if i]
        if num_gpus > 0:
            gpu_ids = gpu_ids[:num_gpus]
        if len(gpu_ids) == 0:
            raise RuntimeError("free-gpu returned no GPU ids")
    except Exception:
        raise RuntimeError("Failed to find free GPUs via free-gpu")
    return gpu_ids


def tensors_to_device(data: TensorTree, device: torch.device) -> TensorTree:
    """Moves a tensor tree (tensor, dict, list, tuple) to ``device``."""
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].to(device)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = value.to(device)
    elif isinstance(data, tuple):
        data = tuple(value.to(device) for value in data)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        raise Exception(f"Unknown data type for {data}")

    return data


def tensors_to_cpu(data: TensorTree) -> TensorTree:
    """Moves a tensor tree (tensor, dict, list, tuple) to CPU."""
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].cpu()
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = value.cpu()
    elif isinstance(data, tuple):
        data = tuple(value.cpu() for value in data)
    elif isinstance(data, torch.Tensor):
        data = data.cpu()
    else:
        raise Exception(f"Unknown data type for {data}")

    return data


def tensors_to_numpy(data: TensorTree) -> NumpyTree:
    """Converts a tensor tree (tensor, dict, list, tuple) to numpy arrays."""
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].cpu().numpy()
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = value.cpu().numpy()
    elif isinstance(data, tuple):
        data = tuple(value.cpu().numpy() for value in data)
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    else:
        raise Exception(f"Unknown data type for {data}")

    return data


def tensors_subset(
    data: Dict[Any, torch.Tensor],
    keys: Sequence[Any],
    device: Optional[torch.device] = None,
    return_dict: bool = False,
) -> Union[Dict[Any, torch.Tensor], Tuple[torch.Tensor, ...]]:
    """Extracts selected keys from a tensor dict and optionally moves to a device."""
    if return_dict:
        data = {k: data[k] for k in keys}
    else:
        data = tuple(data[k] for k in keys)

    if device is not None:
        data = tensors_to_device(data, device)

    return data
