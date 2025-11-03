"""Utility helpers for training and evaluation."""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """Persist model (and optional optimizer) state to *path*.

    Ensures the parent directory exists before writing the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["opt_state"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
):
    """Load checkpoint from *path* into ``model`` and optionally ``optimizer``."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "opt_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt_state"])
    return checkpoint


def ctc_greedy_decode(logits: torch.Tensor, idx2char: Dict[int, str]) -> Iterable[str]:
    """Greedy CTC decode.

    Args:
        logits: Tensor shaped (T, B, C) where T is time steps, B batch size, C classes.
        idx2char: mapping from class indices to characters (0 is treated as blank).

    Returns:
        List of decoded strings for each element in the batch.
    """
    probs = torch.nn.functional.log_softmax(logits, dim=2)
    preds = probs.argmax(2).transpose(0, 1).cpu().numpy()

    results = []
    for seq in preds:
        out = []
        prev = -1
        for p in seq:
            if p != prev and p != 0:
                out.append(idx2char.get(int(p), "?"))
            prev = p
        results.append("".join(out))
    return results
