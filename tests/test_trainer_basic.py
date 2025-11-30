# tests/test_trainer_basic.py
import numpy as np
from multibin_cvae import CVAETrainer

def test_trainer_initialization():
    tr = CVAETrainer(
        x_dim=4,
        y_dim=3,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=16
    )
    assert tr.model is not None
    assert tr.x_dim == 4
    assert tr.y_dim == 3

def test_trainer_forward_shapes():
    tr = CVAETrainer(
        x_dim=4,
        y_dim=3,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=16
    )
    X = np.random.randn(5,4).astype(np.float32)
    logits = tr._forward_logits(tr._standardize(X))
    assert logits.shape == (5,3)
