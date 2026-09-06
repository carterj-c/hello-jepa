"""I-JEPA-style smoke test over the synthetic HMM time series.

This is an infrastructure check, not a modelling result: we are testing whether
a JEPA trains at all on data whose ground truth is known (a logistic regression
on the raw flattened window already gets 0.817 test accuracy). Every epoch
prints instrumentation -- including two collapse metrics -- so a run that
"looks like it worked" but has collapsed cannot hide.

Import is silent and cheap; training only runs under `if __name__ == "__main__"`.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow `uv run python src/jepa.py` (script mode) to resolve `src.*` imports the
# same way `python -m` / `-c` from the repo root would.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.HMM_data import X_train, WINDOW
from src.vicreg_loss import VICRegLoss

N_PATCHES = 8
PATCH_LEN = WINDOW // N_PATCHES          # 8 timesteps per patch
N_CHANNELS = 4
PATCH_DIM = PATCH_LEN * N_CHANNELS       # 32
D_MODEL = 128


def _patchify(x: torch.Tensor) -> torch.Tensor:
    """[B, WINDOW, C] -> [B, N_PATCHES, PATCH_DIM]."""
    b = x.shape[0]
    x = x.reshape(b, N_PATCHES, PATCH_LEN, N_CHANNELS)
    return x.reshape(b, N_PATCHES, PATCH_DIM)


class Encoder(nn.Module):
    """Patchify + linear embed + positional embedding + TransformerEncoder."""

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.patch_embed = nn.Linear(PATCH_DIM, D_MODEL)
        self.pos_embed = nn.Parameter(torch.empty(N_PATCHES, D_MODEL))
        nn.init.trunc_normal_(self.pos_embed, std=0.02, generator=g)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, WINDOW, C] -> patch embeddings [B, N_PATCHES, D_MODEL]."""
        patches = _patchify(x)
        h = self.patch_embed(patches) + self.pos_embed.unsqueeze(0)
        return self.encoder(h)


class Predictor(nn.Module):
    """2-layer transformer predicting target patch embeddings from context."""

    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=2)

    def forward(
        self,
        context: torch.Tensor,
        target_pos_embed: torch.Tensor,
        n_target: int,
    ) -> torch.Tensor:
        """context: [B, n_ctx, D_MODEL], target_pos_embed: [n_target, D_MODEL].

        Returns predictions for the target positions: [B, n_target, D_MODEL].
        """
        b = context.shape[0]
        mask_tokens = self.mask_token.expand(b, n_target, D_MODEL) + target_pos_embed.unsqueeze(0)
        tokens = torch.cat([context, mask_tokens], dim=1)
        out = self.predictor(tokens)
        return out[:, -n_target:, :]


def build_encoder(seed: int = 0) -> nn.Module:
    """Returns an UNTRAINED encoder (used for the random-init control)."""
    torch.manual_seed(seed)
    return Encoder(seed=seed)


def embed(encoder: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Frozen embedding: mean-pool patch embeddings over the 8 positions.

    Returns float64 [N, D_MODEL].
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = np.ascontiguousarray(X[i : i + batch_size]).astype(np.float32)
            t = torch.from_numpy(batch).to(device)
            h = encoder(t)                      # [B, N_PATCHES, D_MODEL]
            pooled = h.mean(dim=1)               # [B, D_MODEL]
            outs.append(pooled.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float64)


@torch.no_grad()
def _update_ema(target: nn.Module, context: nn.Module, momentum: float) -> None:
    for tp, cp in zip(target.parameters(), context.parameters()):
        tp.mul_(momentum).add_(cp, alpha=1.0 - momentum)
    for tb, cb in zip(target.buffers(), context.buffers()):
        tb.copy_(cb)


def _sample_target_block(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Sample one contiguous target block of 2-4 patches; rest is context."""
    block_len = int(rng.integers(2, 5))          # 2..4 inclusive
    start = int(rng.integers(0, N_PATCHES - block_len + 1))
    target_idx = list(range(start, start + block_len))
    context_idx = [i for i in range(N_PATCHES) if i not in target_idx]
    return context_idx, target_idx


@torch.no_grad()
def _collapse_metrics(context_encoder: nn.Module, sample: torch.Tensor) -> tuple[float, float]:
    """emb_std and eff_rank computed on a fixed sample of mean-pooled embeddings."""
    context_encoder.eval()
    h = context_encoder(sample)          # [N, N_PATCHES, D_MODEL]
    pooled = h.mean(dim=1)               # [N, D_MODEL]
    emb_std = pooled.std(dim=0).mean().item()

    centered = pooled - pooled.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    p = (s ** 2)
    p = p / p.sum().clamp_min(1e-12)
    entropy = -(p * (p.clamp_min(1e-12)).log()).sum()
    eff_rank = torch.exp(entropy).item()
    return emb_std, eff_rank


def train_jepa(
    epochs: int = 30,
    seed: int = 0,
    verbose: bool = True,
    inv_coeff: float = 25.0,
    var_coeff: float = 15.0,
    cov_coeff: float = 1.0,
) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    context_encoder = Encoder(seed=seed)
    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    predictor = Predictor()

    loss_fn = VICRegLoss(inv_coeff=inv_coeff, var_coeff=var_coeff, cov_coeff=cov_coeff)

    params = list(context_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)

    X = np.ascontiguousarray(X_train).astype(np.float32)
    n = X.shape[0]
    batch_size = 256
    n_batches = math.ceil(n / batch_size)
    total_steps = epochs * n_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    momentum_start, momentum_end = 0.996, 1.0

    # Fixed held-out-from-training sample of train windows for collapse metrics.
    probe_idx = rng.choice(n, size=min(1024, n), replace=False)
    probe_sample = torch.from_numpy(X[probe_idx])

    metrics_history: list[dict] = []
    step = 0
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = epoch_inv = epoch_var = epoch_cov = 0.0
        n_seen = 0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            batch = torch.from_numpy(X[idx])

            context_idx, target_idx = _sample_target_block(rng)

            momentum = momentum_start + (momentum_end - momentum_start) * (step / max(total_steps - 1, 1))

            context_encoder.train()
            full_ctx = context_encoder(batch)                     # [B, N_PATCHES, D_MODEL]
            context_emb = full_ctx[:, context_idx, :]              # [B, n_ctx, D_MODEL]

            with torch.no_grad():
                target_encoder.eval()
                full_tgt = target_encoder(batch)                   # [B, N_PATCHES, D_MODEL]
                target_emb = full_tgt[:, target_idx, :]            # [B, n_tgt, D_MODEL]

            target_pos = context_encoder.pos_embed[target_idx]     # [n_tgt, D_MODEL]
            pred = predictor(context_emb, target_pos, len(target_idx))  # [B, n_tgt, D_MODEL]

            pred_flat = pred.reshape(-1, D_MODEL)
            target_flat = target_emb.reshape(-1, D_MODEL)
            losses = loss_fn(pred_flat, target_flat)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            scheduler.step()

            _update_ema(target_encoder, context_encoder, momentum)

            bsz = batch.shape[0]
            epoch_loss += losses["loss"].item() * bsz
            epoch_inv += losses["inv-loss"].item() * bsz
            epoch_var += losses["var-loss"].item() * bsz
            epoch_cov += losses["cov-loss"].item() * bsz
            n_seen += bsz
            step += 1

        emb_std, eff_rank = _collapse_metrics(context_encoder, probe_sample)

        record = {
            "epoch": epoch,
            "loss": epoch_loss / n_seen,
            "inv_loss": epoch_inv / n_seen,
            "var_loss": epoch_var / n_seen,
            "cov_loss": epoch_cov / n_seen,
            "momentum": momentum,
            "emb_std": emb_std,
            "eff_rank": eff_rank,
        }
        metrics_history.append(record)

        if verbose:
            print(
                f"epoch {record['epoch']:3d}  "
                f"loss {record['loss']:8.4f}  "
                f"inv {record['inv_loss']:8.4f}  "
                f"var {record['var_loss']:8.4f}  "
                f"cov {record['cov_loss']:8.4f}  "
                f"momentum {record['momentum']:.5f}  "
                f"emb_std {record['emb_std']:8.4f}  "
                f"eff_rank {record['eff_rank']:8.4f}"
            )

    return context_encoder, metrics_history


if __name__ == "__main__":
    train_jepa()
