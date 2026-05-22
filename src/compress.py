"""DeltaKV compression utilities: SVD keys + INT8 values."""

import torch
import numpy as np
from typing import Tuple, Optional


def svd_compress_keys(keys: torch.Tensor, rank: int = 8) -> torch.Tensor:
    """Compress key tensor via SVD across merged head/token dimension.

    Args:
        keys: Tensor of shape (1, H, S, D)
        rank: Number of singular values to keep

    Returns:
        Reconstructed key tensor, same shape as input
    """
    B, H, S, D = keys.shape
    flat = keys[0].reshape(H * S, D).contiguous().float()
    r = min(rank, H * S, D)
    if r < 1:
        return keys

    try:
        U, S_val, V = torch.svd_lowrank(flat, q=r, niter=2)
        recon = (U @ torch.diag(S_val) @ V.mT).reshape(H, S, D).unsqueeze(0)
        return recon.to(keys.dtype)
    except RuntimeError:
        return keys


def int8_compress_values(values: torch.Tensor) -> torch.Tensor:
    """Quantize values to INT8 with per-token scaling.

    Args:
        values: Tensor of shape (1, H, S, D)

    Returns:
        Dequantized values, same shape as input
    """
    v_float = values.float()
    scale = v_float.abs().amax(dim=-1, keepdim=True) + 1e-8
    scaled = (v_float * 127 / scale).round().clamp(-128, 127).char()
    recon = (scaled.float() * scale / 127).to(values.dtype)
    return recon


class DeltaKV:
    """DeltaKV compression manager.

    Combines SVD key compression with INT8 value quantization.
    Provides exact storage accounting.
    """

    @staticmethod
    def compute_storage(
        n_layers: int,
        n_heads: int,
        head_dim: int,
        svd_rank: int = 8,
        use_int8: bool = True,
        prediction_gain: float = 0.0,
    ) -> float:
        """Compute exact storage in bytes per token.

        Args:
            n_layers: Number of attention layers
            n_heads: Number of KV heads
            head_dim: Key/value dimension
            svd_rank: SVD rank for key compression
            use_int8: Whether to use INT8 for values
            prediction_gain: Fractional prediction gain (0.51 = 51%)

        Returns:
            Bytes per token for compressed storage
        """
        T_AMORT = 32  # amortize SVD metadata over 32-token blocks

        # Keys: SVD compressed
        key_u = n_heads * svd_rank  # U matrix per token
        key_s = svd_rank / T_AMORT  # Singular values amortized
        key_vt = svd_rank * head_dim / T_AMORT  # V matrix amortized

        key_bytes = (key_u + key_s + key_vt) * 2 * n_layers  # fp16

        # Apply prediction gain
        if prediction_gain > 0:
            key_bytes *= (1 - prediction_gain)

        # Values: INT8 or FP16
        if use_int8:
            val_bytes = n_heads * head_dim * 1 * n_layers  # int8
        else:
            val_bytes = n_heads * head_dim * 2 * n_layers  # fp16

        return (key_bytes + val_bytes) * 2  # K + V

    @staticmethod
    def summarize(n_layers, n_heads, head_dim, svd_rank=8):
        """Print compressed storage summary."""
        standard = 144 * 1024  # bytes/token for standard 4B
        compressed = DeltaKV.compute_storage(n_layers, n_heads, head_dim, svd_rank)

        print(f"\nDeltaKV Compression Summary:")
        print(f"  {'Component':<25} {'Bytes/tok':<12} {'Ratio':<10}")
        print("-" * 50)
        print(f"  {'Standard attention':<25} {standard:<12,} 1.0x")

        for rank in [8, 16]:
            for pred in [0.0, 0.51]:
                for int8 in [False, True]:
                    label = f"SVD r={rank}"
                    if pred > 0:
                        label += " + pred"
                    if int8:
                        label += " + INT8"
                    b = DeltaKV.compute_storage(n_layers, n_heads, head_dim, rank, int8, pred)
                    print(f"  {label:<25} {b:<12,.0f} {standard/b:>8.1f}x")
