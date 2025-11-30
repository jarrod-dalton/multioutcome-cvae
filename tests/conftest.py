# tests/conftest.py
import numpy as np
import torch
import pytest
import os

@pytest.fixture(autouse=True)
def fixed_seed():
    """
    Ensure deterministic RNG for tests that *can* be deterministic.
    CVAE training is not fully deterministic, but simulation and
    forward passes can be made so.
    """
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    yield
