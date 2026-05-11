"""
MLP that maps RL environment observations to fNIRS feature vectors for use with
the existing sklearn neural decoder (same shape as ``processor.get_fnirs_sample``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# Matches default ``fnirs_channels`` in ``DatasetProcessor.get_fnirs_sample``.
DEFAULT_FNIRS_DIM = 8


class FnirsFeaturePredictor(nn.Module):
    """
    Predicts a vector of fNIRS channel values from a flat state observation.

    Parameters
    ----------
    state_dim :
        Length of the environment observation vector (after any flattening).
    fnirs_dim :
        Number of fNIRS channels in the target vector (default 8).
    hidden_sizes :
        Hidden layer widths for the MLP trunk.
    """

    def __init__(
        self,
        state_dim: int,
        fnirs_dim: int = DEFAULT_FNIRS_DIM,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.fnirs_dim = int(fnirs_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        layers: List[nn.Module] = []
        in_f = self.state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_f, int(h)), nn.ReLU()])
            in_f = int(h)
        layers.append(nn.Linear(in_f, self.fnirs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state :
            Shape ``(batch, state_dim)`` or ``(state_dim,)``.

        Returns
        -------
        Predicted fNIRS vector, shape ``(batch, fnirs_dim)`` or ``(fnirs_dim,)``.
        """
        squeeze = state.dim() == 1
        x = state.float()
        if squeeze:
            x = x.unsqueeze(0)
        out = self.net(x)
        if squeeze:
            out = out.squeeze(0)
        return out

    @torch.inference_mode()
    def predict_numpy(
        self,
        state: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Numpy in/out helper for piping predictions into sklearn ``MLPClassifier`` /
        ``MLPRegressor`` (expects shape ``(1, fnirs_dim)`` or ``(fnirs_dim,)``).
        """
        dev = device or next(self.parameters()).device
        s = torch.as_tensor(state, dtype=torch.float32, device=dev)
        y = self.forward(s)
        return y.detach().cpu().numpy().astype(np.float32, copy=False)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "state_dim": self.state_dim,
                "fnirs_dim": self.fnirs_dim,
                "hidden_sizes": self.hidden_sizes,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Tuple["FnirsFeaturePredictor", dict]:
        """
        Load weights and metadata written by ``save``.

        Returns
        -------
        model, checkpoint
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hidden = ckpt.get("hidden_sizes", (256, 256))
        model = cls(
            state_dim=int(ckpt["state_dim"]),
            fnirs_dim=int(ckpt["fnirs_dim"]),
            hidden_sizes=hidden,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt


def build_fnirs_predictor(
    state_dim: int,
    fnirs_dim: int = DEFAULT_FNIRS_DIM,
    hidden_sizes: Sequence[int] = (256, 256),
    device: Optional[torch.device] = None,
) -> FnirsFeaturePredictor:
    """Construct predictor and move to ``device`` (default: CUDA if available)."""
    m = FnirsFeaturePredictor(state_dim, fnirs_dim, hidden_sizes)
    dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return m.to(dev)
