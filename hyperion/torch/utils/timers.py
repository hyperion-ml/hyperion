"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch


class CUDATimer:
    """Utility to measure elapsed CUDA time for named code sections.

    This timer uses ``torch.cuda.Event`` pairs per section name. Call
    :meth:`start`, run your CUDA work, call :meth:`stop`, and then call
    :meth:`synchronize_and_report` to retrieve elapsed times in milliseconds.
    """

    def __init__(self, device=None):
        """Create a CUDA timer.

        Args:
            device: CUDA device identifier accepted by ``torch.device``.
                If ``None``, defaults to the current ``"cuda"`` device.
        """
        self.device = torch.device(device) if device else torch.device("cuda")
        self.events = {}

    def start(self, name: str):
        """Start timing a named section.

        Args:
            name: Section identifier used to store the start/end events.
        """
        with torch.cuda.device(self.device):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            self.events[name] = (s, e)

    def stop(self, name: str):
        """Stop timing a previously started section.

        Args:
            name: Section identifier passed to :meth:`start`.
        """
        _, e = self.events[name]
        with torch.cuda.device(self.device):
            e.record()

    def synchronize_and_report(self):
        """Synchronize recorded events and return elapsed times.

        Returns:
            Dictionary mapping section names to elapsed time in milliseconds.
        """
        out = {}
        for name, (s, e) in self.events.items():
            e.synchronize()
            out[name] = s.elapsed_time(e)
        return out
