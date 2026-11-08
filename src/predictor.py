#!/usr/bin/env python3
"""DeltaFormer: Temporal Predictive Coding for Transformer KV States.

Core predictor module. Provides the KeyPredictor class that learns
to predict KV key states from their temporal predecessors.

Usage:
    from src.predictor import KeyPredictor, train_predictor

    predictor = train_predictor(model, tokenizer, layer_idx=3)
    gain = predictor.evaluate(model, tokenizer, layer_idx=3)
    print(f'Prediction gain: {gain:.1%}')
"""

import torch
import numpy as np
from torch import nn
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class KeyPredictor(nn.Module):
    """Autoregressive key predictor using linear model.

    Predicts k_t from k_{t-2} and k_{t-1}:
        k_hat_t = W * [k_{t-2} || k_{t-1}] + b

    Args:
        head_dim: Dimension of key vectors (e.g., 64, 128, 256)
    """

    def __init__(self, head_dim: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(2 * head_dim, head_dim)

    def forward(self, k_prev2: torch.Tensor, k_prev1: torch.Tensor) -> torch.Tensor:
        """Predict k_t from two previous key vectors.

        Args:
            k_prev2: Key vector at t-2, shape (..., head_dim)
            k_prev1: Key vector at t-1, shape (..., head_dim)

        Returns:
            Predicted key vector at t, shape (..., head_dim)
        """
        x = torch.cat([k_prev2, k_prev1], dim=-1)
        return self.linear(x)

    def compress(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress keys by storing only residual from prediction.

        Args:
            keys: Full key sequence, shape (H, S, D) or (S, D)

        Returns:
            Tuple of (first_2_keys, residuals) where residuals are
            the unpredictable component after subtracting prediction.
        """
        if keys.dim() == 3:
            H, S, D = keys.shape
            flat = keys.reshape(-1, D)
        else:
            flat = keys

        first_2 = flat[:2]
        predictions = self(flat[:-2], flat[1:-1])
        targets = flat[2:]
        residuals = targets - predictions
        return first_2, residuals

    def decompress(self, first_2: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """Reconstruct keys from first two + residuals.

        Args:
            first_2: First two key vectors, shape (2, D)
            residuals: Residual vectors, shape (S-2, D)

        Returns:
            Reconstructed key sequence, shape (S, D)
        """
        recon = [first_2[0:1], first_2[1:2]]
        for i in range(residuals.shape[0]):
            pred = self(recon[-2], recon[-1])
            recon.append(pred + residuals[i:i+1])
        return torch.cat(recon, dim=0)

    @property
    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def train_predictor(
    model,
    tokenizer,
    attn_layers: List[int],
    head_dim: int = 256,
    n_epochs: int = 30,
    lr: float = 5e-4,
    device: str = "cuda",
) -> Dict[int, KeyPredictor]:
    """Train key predictors for specified attention layers.

    Collects ~2000 key entries per layer from dummy text,
    then trains a linear AR(2) predictor.

    Args:
        model: Loaded HuggingFace model (eval mode)
        tokenizer: Corresponding tokenizer
        attn_layers: List of layer indices with standard attention
        head_dim: Dimension of key vectors
        n_epochs: Training epochs per layer
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        Dict mapping layer index to trained KeyPredictor
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    logger.info(f"Training predictors for {len(attn_layers)} layers...")

    # Collect training data
    texts = [
        "The transformer uses attention to process data. " * 15,
        "Machine learning learns patterns from data. " * 15,
    ]
    all_keys = {l: [] for l in attn_layers}

    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        kv = outputs.past_key_values
        for l in attn_layers:
            if hasattr(kv, 'layers'):
                k = kv.layers[l].keys.float()
            else:
                k = kv[l][0].float()
            all_keys[l].append(k[0].transpose(0, 1).reshape(-1, head_dim).cpu())

    # Train per layer
    predictors = {}
    for l in attn_layers:
        keys = torch.cat(all_keys[l])
        k0, k1, k2 = keys[:-2].to(device), keys[1:-1].to(device), keys[2:].to(device)

        pred = KeyPredictor(head_dim).to(device)
        opt = torch.optim.AdamW(pred.parameters(), lr=lr)

        for epoch in range(n_epochs):
            opt.zero_grad()
            loss = nn.MSELoss()(pred(k0, k1), k2)
            loss.backward()
            opt.step()

        with torch.no_grad():
            gain = 1 - (k2 - pred(k0, k1)).norm(dim=-1).mean() / k2.norm(dim=-1).mean()

        predictors[l] = pred.eval()
        logger.info(f"  Layer {l}: gain={gain.item()*100:.0f}% (params={pred.parameter_count})")

    return predictors


def compute_compression_ratio(
    n_attn_layers: int,
    n_kv_heads: int,
    head_dim: int,
    svd_rank: int = 8,
    use_int8: bool = True,
    prediction_gain: float = 0.0,
) -> float:
    """Compute compression ratio from component parameters.

    Args:
        n_attn_layers: Number of attention layers
        n_kv_heads: Number of KV heads
        head_dim: Key/value dimension
        svd_rank: SVD compression rank
        use_int8: Whether to use INT8 for values
        prediction_gain: Fractional key prediction gain (0.51)

    Returns:
        Compression ratio vs standard 4B model (144 KB/token)
    """
    from src.compress import DeltaKV

    baseline = 144 * 1024  # bytes per token for standard 4B
    compressed = DeltaKV.compute_storage(
        n_attn_layers, n_kv_heads, head_dim, svd_rank, use_int8, prediction_gain
    )
    return baseline / compressed


def fit_ar2_least_squares(keys: torch.Tensor) -> KeyPredictor:
    """Fit a KeyPredictor in closed form (least squares) on a key sequence.

    Args:
        keys: Key sequence of shape (S, D) for one head, in token order.

    Returns:
        KeyPredictor whose weights minimise squared error of k_t given
        [k_{t-2} || k_{t-1}] over this sequence.
    """
    keys = keys.float()
    x = torch.cat([keys[:-2], keys[1:-1], torch.ones(keys.shape[0] - 2, 1)], dim=-1)
    y = keys[2:]
    sol = torch.linalg.lstsq(x, y).solution  # (2D + 1, D)
    d = keys.shape[1]
    pred = KeyPredictor(d)
    with torch.no_grad():
        pred.linear.weight.copy_(sol[:-1].T)
        pred.linear.bias.copy_(sol[-1])
    return pred.eval()


def prediction_gain(pred: KeyPredictor, keys: torch.Tensor) -> float:
    """Fraction of key norm removed by prediction: 1 - |residual| / |key|."""
    keys = keys.float()
    with torch.no_grad():
        res = keys[2:] - pred(keys[:-2], keys[1:-1])
    return float(1 - res.norm(dim=-1).mean() / keys[2:].norm(dim=-1).mean())


def forward_split_gain(keys: torch.Tensor, train_frac: float = 0.5) -> Dict[str, float]:
    """Held-out, forward-in-time evaluation of the AR(2) key predictor.

    Fits on the first ``train_frac`` of the tokens and scores on the rest,
    so no later token is used to fit the model it is scored with. Returns
    both the in-sample gain and the held-out gain; a large gap between them
    means the in-sample number overstates how predictable the keys are.

    Args:
        keys: Key sequence of shape (S, D) for one head, in token order.
        train_frac: Fraction of tokens (from the start) used for fitting.

    Returns:
        {"in_sample": float, "held_out": float}
    """
    split = int(keys.shape[0] * train_frac)
    if split < 4 or keys.shape[0] - split < 3:
        raise ValueError("sequence too short for a forward split")
    pred = fit_ar2_least_squares(keys[:split])
    return {
        "in_sample": prediction_gain(pred, keys[:split]),
        "held_out": prediction_gain(pred, keys[split:]),
    }
