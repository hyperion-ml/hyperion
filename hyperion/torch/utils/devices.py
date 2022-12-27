"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import subprocess
import logging

import torch


def open_device(num_gpus=1, gpu_ids=None, find_free_gpu=False):
    if find_free_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu_ids is None:
            gpu_ids = find_free_gpus(num_gpus)
        if isinstance(gpu_ids, list):
            gpu_ids = ",".join([str(i) for i in gpu_ids])

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    if num_gpus > 0 and torch.cuda.is_available():
        logging.info("CUDA_VISIBLE_DEVICES=%s" % os.environ["CUDA_VISIBLE_DEVICES"])
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


def find_free_gpus(num_gpus):
    try:
        result = subprocess.run("free-gpu", stdout=subprocess.PIPE)
        gpu_ids = result.stdout.decode("utf-8")
    except:
        gpu_ids = "0"
    return gpu_ids


def tensors_to_device(data, device):
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].to(device)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = value.to(device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        raise Exception(f"Unknown data type for {data}")

    return data


def tensors_to_cpu(data):
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].cpu()
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = value.cpu()
    elif isinstance(data, torch.Tensor):
        data = data.cpu()
    else:
        raise Exception(f"Unknown data type for {data}")

    return data


def tensors_to_numpy(data):
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].cpu().numpy()
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = value.cpu().numpy()
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    else:
        raise Exception(f"Unknown data type for {data}")

    return data
