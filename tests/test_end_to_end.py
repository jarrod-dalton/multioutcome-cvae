# tests/test_end_to_end.py
from multioutcome_cvae import simulate_cvae_data, CVAETrainer
import numpy as np

def test_end_to_end_small_model():
    X, Y, _ = simulate_cvae_data(
        n_samples=500,
        n_features=4,
        n_outcomes=5,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=777,
    )

    tr = CVAETrainer(
        x_dim=4,
        y_dim=5,
        latent_dim=3,
        hidden_dim=32,
        n_hidden_layers=1,
        outcome_type="bernoulli",
    )

    hist = tr.fit(X,Y,epochs=5,verbose=False)

    # Loss should decrease somewhat (not strict)
    assert len(hist["train_loss"]) == 5
    assert hist["train_loss"][0] > hist["train_loss"][-1]

    # Generate samples
    Y_sim = tr.generate(X[:5],n_samples_per_x=5)
    assert Y_sim.shape == (5,5,5)
