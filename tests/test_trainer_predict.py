# tests/test_trainer_predict.py
import numpy as np
from multioutcome_cvae import simulate_cvae_data, CVAETrainer

def test_predict_proba_in_range():
    X, Y, _ = simulate_cvae_data(
        n_samples=500,
        n_features=3,
        n_outcomes=5,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=123,
    )
    tr = CVAETrainer(3,5,latent_dim=4,outcome_type="bernoulli")
    tr.fit(X,Y,epochs=3,verbose=False)
    p = tr.predict_proba(X[:10],n_mc=5)
    assert p.shape == (10,5)
    assert np.all(p >= 0) and np.all(p <= 1)

def test_generate_shape():
    X, Y, _ = simulate_cvae_data(
        n_samples=200,
        n_features=3,
        n_outcomes=4,
        latent_dim=1,
        outcome_type="bernoulli",
        seed=123,
    )
    tr = CVAETrainer(3,4,latent_dim=2,outcome_type="bernoulli")
    tr.fit(X,Y,epochs=2,verbose=False)
    Y_sim = tr.generate(X[:5], n_samples_per_x=3)
    assert Y_sim.shape == (5,3,4)


def test_inference_batch_size_bounds_decoder_batches():
    X, Y, _ = simulate_cvae_data(
        n_samples=30,
        n_features=3,
        n_outcomes=4,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=456,
    )
    tr = CVAETrainer(3, 4, latent_dim=2, outcome_type="bernoulli")
    tr.fit(X, Y, epochs=1, verbose=False, seed=456)

    original_decode = tr.model.decode
    decoder_batch_sizes = []

    def recording_decode(x, z):
        decoder_batch_sizes.append(x.shape[0])
        return original_decode(x, z)

    tr.model.decode = recording_decode

    probs = tr.predict_proba(X[:10], n_mc=3, inference_batch_size=3)
    assert probs.shape == (10, 4)
    assert max(decoder_batch_sizes) <= 3

    decoder_batch_sizes.clear()
    samples = tr.generate(
        X[:5], n_samples_per_x=3, inference_batch_size=2
    )
    assert samples.shape == (5, 3, 4)
    assert max(decoder_batch_sizes) <= 2 * 3
