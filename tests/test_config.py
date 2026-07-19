from config import get_config, get_device, set_seed
from config.settings import Config
import torch


def test_get_config_loads_default():
    cfg = get_config()
    assert isinstance(cfg, Config)


def test_config_attribute_access(cfg):
    assert cfg.training.gamma == 0.99
    assert cfg.device == "auto"


def test_config_nested_access(cfg):
    assert cfg.data.embed_dir == "data_3_embed"
    assert cfg.etl.chunk_size == 50


def test_get_device_auto_returns_device(cfg):
    device = get_device(cfg)
    assert isinstance(device, torch.device)


def test_set_seed_deterministic(cfg):
    set_seed(cfg)
    a = torch.rand(5)
    set_seed(cfg)
    b = torch.rand(5)
    assert torch.equal(a, b)
