"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math

import torch.distributed as dist


class GradNormTracker:
    """Tracks the Exponential Moving Average (EMA) of gradients and detects spikes.
    This class maintains an EMA for each parameter's gradient and checks if the current
    gradient exceeds the EMA by a specified spike threshold.
    Attributes:
        ema (dict): Dictionary to store EMA of gradients for each parameter.
        decay (float): Decay factor for EMA.
        spike_threshold (float): Threshold to detect spikes in gradients.
    """

    def __init__(self, decay=0.99, spike_threshold=10):
        self.ema = {}  # Stores EMA per parameter name
        self.spikes = {}
        self.decay = decay
        self.spike_threshold = spike_threshold
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size

    @property
    def grad_ema(self) -> dict:
        """Returns the current EMA of gradients."""
        return {f"{k}_ema": v for k, v in self.ema.items() if v is not None}

    @property
    def grad_spikes(self) -> dict:
        """Returns the current spikes detected in gradients."""
        return {f"{k}_spike": v for k, v in self.spikes.items() if v is not None}

    def update(self, current: dict) -> bool:
        """Update the EMA of gradients and check for spikes.
        Args:
            current (dict): Current gradients with parameter names as keys.
        Returns:
            bool: True if a spike is detected, False otherwise.
        """
        if self.rank != 0:
            return False
        spike_detected = False
        for name, value in current.items():
            if math.isnan(value) or math.isinf(value):
                spike_detected = True
                self.spikes[name] = value
                continue

            if name not in self.ema:
                self.ema[name] = value
            else:
                self.ema[name] = self.decay * self.ema[name] + (1 - self.decay) * value

            if value > self.ema[name] * self.spike_threshold:
                # logging.warning(f"⚠️ Gradient spike detected for {name}: {value:.4f} (EMA: {self.ema[name]:.4f})")
                spike_detected = True
                self.spikes[name] = value
            else:
                self.spikes[name] = None

        return spike_detected
