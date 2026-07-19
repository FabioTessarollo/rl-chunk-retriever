import os
import random
import yaml
import torch
import numpy as np


class Config:
    """Nested attribute-access wrapper around a dict loaded from YAML."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({vars(self)})"


_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_config(path: str = _DEFAULT_PATH) -> Config:
    """Load configuration from a YAML file and return a Config object."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def get_device(cfg: Config) -> torch.device:
    """Resolve device from config. 'auto' picks the best available."""
    device_str = cfg.device
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def set_seed(cfg: Config) -> None:
    """Set random seeds for reproducibility."""
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
