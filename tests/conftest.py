import sys
import os
import pytest
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import get_config, get_device, set_seed
from config.settings import Config


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def device(cfg):
    return get_device(cfg)
