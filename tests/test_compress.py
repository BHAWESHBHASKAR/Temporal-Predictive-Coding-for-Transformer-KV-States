"""CPU-only tests for SVD key compression, INT8 values and storage accounting."""

import torch

from src.compress import DeltaKV, int8_compress_values, svd_compress_keys


def test_int8_values_keep_shape_and_bounded_error():
    torch.manual_seed(0)
    v = torch.randn(1, 2, 32, 16)
    r = int8_compress_values(v)
    assert r.shape == v.shape
    # per-token scale: error at most half a quantisation step
    step = v.abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((r - v).abs() <= step / 2 + 1e-6)


def test_svd_keys_exact_for_low_rank_input():
    torch.manual_seed(0)
    a = torch.randn(2 * 32, 4)
    b = torch.randn(4, 16)
    keys = (a @ b).reshape(1, 2, 32, 16)  # rank 4 across merged heads/tokens
    r = svd_compress_keys(keys, rank=8)
    assert r.shape == keys.shape
    assert torch.allclose(r, keys, atol=1e-3)


def test_storage_falls_with_prediction_gain_and_int8():
    base = DeltaKV.compute_storage(6, 2, 256, svd_rank=8, use_int8=False)
    with_int8 = DeltaKV.compute_storage(6, 2, 256, svd_rank=8, use_int8=True)
    with_pred = DeltaKV.compute_storage(6, 2, 256, svd_rank=8, use_int8=True, prediction_gain=0.5)
    assert with_int8 < base
    assert with_pred < with_int8
