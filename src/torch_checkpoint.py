"""Load torch checkpoints saved with full pickle (e.g. numpy scalars in dicts).

PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
many older / locally saved files. Use this for trusted project checkpoints.
"""
from __future__ import annotations

import torch


def torch_load_checkpoint(path: str, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        # PyTorch < 2.0 has no weights_only
        return torch.load(path, **kwargs)
