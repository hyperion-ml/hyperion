"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch


class CUDATimer:
    def __init__(self, device=None):
        self.device = torch.device(device) if device else torch.device("cuda")
        self.events = {}

    def start(self, name: str):
        with torch.cuda.device(self.device):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            self.events[name] = (s, e)

    def stop(self, name: str):
        _, e = self.events[name]
        with torch.cuda.device(self.device):
            e.record()

    def synchronize_and_report(self):
        out = {}
        for name, (s, e) in self.events.items():
            e.synchronize()
            out[name] = s.elapsed_time(e)
        return out
