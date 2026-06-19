"""
Torch device management utilities.

Duplicated from napistu-torch to avoid a circular dependency (napistu-torch
imports napistu-py). Only the subset used by network propagation is included.

Public Functions
---------------
empty_cache(device)
    Empty the cache for a given device.
ensure_device(device, allow_autoselect, mps_valid)
    Coerce a device argument to torch.device, with optional auto-selection.
memory_manager(device)
    Context manager that clears device cache before and after an operation.
select_device(mps_valid)
    Auto-select the best available device: MPS > CUDA > CPU.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional, Union

from napistu.utils.constants import DEVICES

logger = logging.getLogger(__name__)


def empty_cache(device) -> None:
    """Empty the cache for a given device. No-op for CPU."""
    import torch

    if isinstance(device, str):
        device = torch.device(device)
    if device.type == DEVICES.MPS and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.type == DEVICES.CUDA and torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_device(
    device: Optional[Union[str, "torch.device"]],
    allow_autoselect: bool = False,
    mps_valid: bool = True,
) -> "torch.device":
    """Coerce device to torch.device, optionally auto-selecting if None."""
    import torch

    if device is None:
        if allow_autoselect:
            return select_device(mps_valid=mps_valid)
        raise ValueError("An explicit device is required but was not specified")
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    raise ValueError(f"Invalid device: {device!r}, must be a string or torch.device")


@contextmanager
def memory_manager(device=None):
    """Context manager that clears device cache before and after an operation."""
    import torch

    if device is None:
        device = torch.device(DEVICES.CPU)
    empty_cache(device)
    try:
        yield
    finally:
        empty_cache(device)
        gc.collect()


def select_device(mps_valid: bool = True) -> "torch.device":
    """Select the best available device: MPS > CUDA > CPU."""
    import torch

    if mps_valid and torch.backends.mps.is_available():
        return torch.device(DEVICES.MPS)
    if torch.cuda.is_available():
        return torch.device(DEVICES.CUDA)
    return torch.device(DEVICES.CPU)
