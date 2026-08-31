import logging
import os
import random

import numpy as np
import torch
import yaml


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


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logger with console and optional file handler."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Root must allow DEBUG through so the file handler can capture verbose
    # logs even when the console handler is limited to a higher level.
    root.setLevel(logging.DEBUG if log_file else log_level)

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING if log_file else log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for name in (
        "transformers",
        "sentence_transformers",
        "matplotlib",
        "PIL",
        "git",
        "git.cmd",
        "git.util",
        "urllib3",
        "mlflow",
        "mlflow_skinny",
        "mlflow_tracing",
        "docker",
        "azure",
        "botocore",
        "boto3",
        "opentelemetry",
        "databricks",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
