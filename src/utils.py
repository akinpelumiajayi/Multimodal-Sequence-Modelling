import os
import random
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed: int = 42):
  """
  For ease reproducibility and generating same random number of sequence.
  """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
  """
  To determine unit availability for operating
  """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype(device: str):
  """
  For proper assigning of torch tensor to avoid mismatch tensor errors
  """
    return torch.float16 if device == "cuda" else torch.float32

def maybe_drop_condition(cond: torch.Tensor, p: float):
  """
    Applies classifier-free guidance style conditioning dropout during training.
    With probability p, replaces conditioning tokens with zeros, which teaches
    the model to also operate in an unconditional mode. This enables CFG at
    inference time.
    Args:
        cond: Conditioning tensor (B, T, D).
        p: Drop probability in [0,1].
    Returns:
        Conditionally dropped tensor of same shape as `cond`.
    """
    return torch.zeros_like(cond) if random.random() < p else cond

def to_minus1_1(x):  # x in [0,1]
  """
  To normalize and ensure the latents are within the range of 0, and 1.
  """
    return x * 2.0 - 1.0

def save_checkpoint(path, payload: dict):
  """
  To save the model weights and other parameters assigned to it.
  """
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)

def load_yaml(path: str):
  """
  To load the configuration for production
  """
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_paths(cfg: dict):
  """
  For ease navigation during training or other necessary functions.
  """
    root = cfg["paths"]["root"]
    ckpt_dir = os.path.join(root, cfg["paths"]["ckpt_dir"])
    results_dir = os.path.join(root, cfg["paths"]["results_dir"])
    ensure_dir(ckpt_dir)
    ensure_dir(results_dir)
    return root, ckpt_dir, results_dir
