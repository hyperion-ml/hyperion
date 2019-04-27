"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import subprocess
import logging

import torch

def open_device(num_gpus=1, gpu_ids=None):
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    if gpu_ids is None:
        try:
            result = subprocess.run('free-gpu', stdout=subprocess.PIPE)
            gpu_ids = result.stdout.decode('utf-8')
        except:
            gpu_ids = '0'

    if isinstance(gpu_ids, list):
        gpu_ids = ','.join([str(i) for i in gpu_ids])
        
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    logging.info('CUDA_VISIBLE_DEVICES=%s' % os.environ['CUDA_VISIBLE_DEVICES'])
    if  torch.cuda.is_available():
        device = torch.device('cuda')
        torch.tensor([0]).to(device)
    else:
        device = torch.device('cpu')
    return device


