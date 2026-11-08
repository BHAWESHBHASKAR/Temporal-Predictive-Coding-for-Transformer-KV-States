"""CPU-only tests for the AR(2) key predictor (no model download needed)."""

import pytest
import torch

from src.predictor import (
    KeyPredictor,
    fit_ar2_least_squares,
    forward_split_gain,
    prediction_gain,
)


def ar2_sequence(n: int, d: int, noise: float = 0.01, seed: int = 0) -> torch.Tensor:
    """Stationary AR(2) oscillator: k_t = 1.9 k_{t-1} - 0.99 k_{t-2} + noise.

    Close to the unit circle, so the signal is large relative to the noise
    and about 96% of each key's norm is predictable from the two before it.
    """
    g = torch.Generator().manual_seed(seed)
    k = [torch.randn(d, generator=g), torch.randn(d, generator=g)]
    for _ in range(n - 2):
        k.append(1.9 * k[-1] - 0.99 * k[-2] + noise * torch.randn(d, generator=g))
    return torch.stack(k)


def test_forward_shape():
    pred = KeyPredictor(head_dim=16)
    out = pred(torch.randn(5, 16), torch.randn(5, 16))
    assert out.shape == (5, 16)


def test_parameter_count():
    assert KeyPredictor(head_dim=8).parameter_count == 2 * 8 * 8 + 8


def test_compress_decompress_round_trip():
    torch.manual_seed(0)
    pred = KeyPredictor(head_dim=16).eval()
    keys = torch.randn(40, 16)
    with torch.no_grad():
        first_2, residuals = pred.compress(keys)
        recon = pred.decompress(first_2, residuals)
    assert residuals.shape == (38, 16)
    assert torch.allclose(recon, keys, atol=1e-4)


def test_least_squares_recovers_ar2_process():
    keys = ar2_sequence(400, 8)
    pred = fit_ar2_least_squares(keys)
    assert prediction_gain(pred, keys) > 0.85


def test_held_out_gain_survives_on_a_truly_predictable_series():
    gains = forward_split_gain(ar2_sequence(400, 8), train_frac=0.5)
    assert gains["held_out"] > 0.85


def test_in_sample_gain_overstates_noise():
    # White noise has nothing to predict. With few tokens per parameter the
    # least-squares fit still "finds" structure in-sample, but the forward,
    # held-out score shows it does not carry over to later tokens.
    torch.manual_seed(0)
    noise = torch.randn(80, 16)
    gains = forward_split_gain(noise, train_frac=0.5)
    assert gains["in_sample"] > 0.3
    assert gains["held_out"] < 0.05
    assert gains["in_sample"] - gains["held_out"] > 0.3


def test_forward_split_rejects_short_sequences():
    with pytest.raises(ValueError):
        forward_split_gain(torch.randn(5, 4))
