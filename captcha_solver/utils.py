"""Utility helpers: saving/loading checkpoints and decoding CTC outputs."""
import torch
import os


def save_checkpoint(path, model, optimizer=None, epoch=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'model_state': model.state_dict()}
    if optimizer is not None:
        state['opt_state'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if extra is not None:
        state['extra'] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    return ckpt


def ctc_greedy_decode(logits, idx2char):
    """Greedy decode logits (T, B, C) -> list of strings"""
    import torch
    soft = torch.nn.functional.log_softmax(logits, dim=2)
    preds = soft.argmax(2)  # T, B
    preds = preds.transpose(0, 1).cpu().numpy()
    results = []
    for seq in preds:
        out = []
        prev = -1
        for p in seq:
            if p != prev and p != 0:
                out.append(idx2char.get(int(p), '?'))
            prev = p
        results.append(''.join(out))
    return results
