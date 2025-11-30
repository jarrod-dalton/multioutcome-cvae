# tests/test_bayeslik.py
import numpy as np
import torch
from multibin_cvae.model import _bernoulli_loglik, _gaussian_loglik, _poisson_loglik

def test_bernoulli_loglik_matches_pytorch():
    y = torch.tensor([[0.,1.,1.],[1.,0.,1.]])
    logits = torch.tensor([[0.1,-1.2,2.3],[1.1,-0.5,0.7]])
    probs = torch.sigmoid(logits)
    dist = torch.distributions.Bernoulli(probs=probs)
    expected = dist.log_prob(y).sum().item()
    got = _bernoulli_loglik(y, logits).item()
    assert abs(expected - got) < 1e-6

def test_gaussian_loglik_manual():
    y = torch.tensor([[0.5, -1.0]])
    mu = torch.tensor([[0.0, 0.0]])
    sigma = torch.tensor([[1.0, 2.0]])
    var = sigma**2
    expected = -0.5 * (((y-mu)**2)/var + 2*torch.log(sigma) + np.log(2*np.pi)).sum().item()
    got = _gaussian_loglik(y, mu, sigma).item()
    assert abs(expected - got) < 1e-6

def test_poisson_loglik_manual():
    y = torch.tensor([[3.,1.]])
    rate = torch.tensor([[2.5, 0.7]])
    expected = (y * torch.log(rate) - rate - torch.lgamma(y+1)).sum().item()
    got = _poisson_loglik(y, rate).item()
    assert abs(expected - got) < 1e-6
