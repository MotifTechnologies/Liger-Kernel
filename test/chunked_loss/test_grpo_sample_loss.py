"""Tests for loss_type="grpo_sample" (per-sample / per-trajectory GRPO).

The fused kernel computes the per-sample loss as a plain weighted sum
``(per_token_loss * sample_weight).sum()`` where the caller bakes
``sample_weight[t] = mask[t] / max(N_{s(t)}, min_tokens)``. We cross-check that
weighted-sum form against an INDEPENDENT scatter_add reference, and anchor a
single-sample case to the existing ``bnpo`` reduction.

Trick: with ``old_per_token_logps=None`` the kernel sets ``old = curr.detach()``
so the ratio is exactly 1 and ``per_token_loss == -advantages`` (no clip/KL/TIS).
That makes the reduction comparable in closed form without re-deriving the PG math.

CPU-only (chunked_loss is pure torch). Run:
    pytest test/chunked_loss/test_grpo_sample_loss.py
"""

import pytest
import torch

from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

DT = torch.double


def _sample_ids(cu, T):
    cu = torch.tensor(cu).long()
    seg = cu[1:] - cu[:-1]
    ids = torch.repeat_interleave(torch.arange(seg.numel()), seg)
    if int(cu[-1]) < T:
        ids = torch.cat([ids, torch.full((T - int(cu[-1]),), -1, dtype=torch.long)])
    return ids[:T].view(1, T)


def _weights(ids, mask, min_tokens):
    """w_t = mask_t / max(N_{s(t)}, min_tokens) — the kernel's per-token weight."""
    flat = ids.reshape(-1)
    valid = (mask.reshape(-1) > 0) & (flat >= 0)
    w = torch.zeros_like(mask, dtype=DT).reshape(-1)
    vid = flat[valid]
    n = int(vid.max()) + 1
    cnt = torch.zeros(n, dtype=torch.long).scatter_add(0, vid, torch.ones_like(vid))
    w[valid] = 1.0 / cnt.clamp(min=min_tokens).to(DT)[vid]
    return w.view_as(mask)


def _ref_sum(per_token_loss, mask, ids, min_tokens):
    """Independent scatter_add reference: sum_s( sum_{t in s} pt_t / max(N_s, min) )."""
    flat_l = per_token_loss.reshape(-1)
    flat_i = ids.reshape(-1)
    valid = (mask.reshape(-1) > 0) & (flat_i >= 0)
    vi, vl = flat_i[valid], flat_l[valid]
    n = int(vi.max()) + 1
    s = torch.zeros(n, dtype=vl.dtype).scatter_add(0, vi, vl)
    c = torch.zeros(n, dtype=torch.long).scatter_add(0, vi, torch.ones_like(vi))
    return (s / c.clamp(min=min_tokens).to(vl.dtype)).sum()


def _fused(loss_type, h, W, sel, mask, adv, sample_weight=None, chunk_size=1):
    fn = LigerFusedLinearGRPOLoss(
        beta=0.0, compiled=False, use_ref_model=False,
        epsilon_low=0.2, epsilon_high=0.2, loss_type=loss_type, chunk_size=chunk_size,
    )
    out = fn(h, W, sel, mask, adv, sample_weight=sample_weight)
    return out[0] if isinstance(out, (tuple, list)) else out


@pytest.mark.parametrize(
    "cu,min_tokens,seed",
    [([0, 3, 7, 10], 2, 1), ([0, 10], 4, 2), ([0, 1, 10], 5, 3), ([0, 5, 10], 1, 4)],
)
def test_grpo_sample_matches_scatter_reference(cu, min_tokens, seed):
    torch.manual_seed(seed)
    B, T, D, V = 1, 10, 8, 16
    h = torch.randn(B, T, D, dtype=DT, requires_grad=True)
    W = torch.randn(V, D, dtype=DT)
    sel = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=DT)
    adv = torch.randn(B, T, dtype=DT)
    ids = _sample_ids(cu, T)

    fused = _fused("grpo_sample", h, W, sel, mask, adv, sample_weight=_weights(ids, mask, min_tokens))
    # ratio == 1 -> per_token_loss == -adv
    ref = _ref_sum(-adv, mask, ids, min_tokens)
    assert torch.allclose(fused.double(), ref.double(), rtol=1e-7, atol=1e-9)


def test_grpo_sample_single_sample_equals_bnpo():
    # One sample, N >= min_tokens, uniform weight 1/N -> grpo_sample == bnpo.
    torch.manual_seed(5)
    B, T, D, V = 1, 12, 8, 16
    h = torch.randn(B, T, D, dtype=DT, requires_grad=True)
    W = torch.randn(V, D, dtype=DT)
    sel = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=DT)
    adv = torch.randn(B, T, dtype=DT)
    ids = _sample_ids([0, T], T)
    w = _weights(ids, mask, min_tokens=4)        # N=12 >= 4 -> w = 1/12

    g_sample = _fused("grpo_sample", h, W, sel, mask, adv, sample_weight=w)
    g_bnpo = _fused("bnpo", h, W, sel, mask, adv)
    assert torch.allclose(g_sample.double(), g_bnpo.double(), rtol=1e-7, atol=1e-9)


def test_grpo_sample_requires_sample_weight():
    torch.manual_seed(6)
    B, T, D, V = 1, 6, 8, 16
    h = torch.randn(B, T, D, dtype=DT, requires_grad=True)
    W = torch.randn(V, D, dtype=DT)
    sel = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=DT)
    adv = torch.randn(B, T, dtype=DT)
    with pytest.raises(AssertionError):
        _fused("grpo_sample", h, W, sel, mask, adv, sample_weight=None)


def test_grpo_sample_multiple_chunks_match_single():
    # B > chunk_size -> chunks > 1: sample_weight is sliced per chunk, so the
    # chunked result must equal the single-chunk result (loss accumulates across
    # chunks) and the scatter reference. Each row is its own trajectory.
    torch.manual_seed(8)
    B, T, D, V = 4, 6, 8, 16
    h = torch.randn(B, T, D, dtype=DT, requires_grad=True)
    W = torch.randn(V, D, dtype=DT)
    sel = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T, dtype=DT)
    for i, n in enumerate([6, 4, 2, 5]):       # varied valid lengths per row
        mask[i, :n] = 1.0
    adv = torch.randn(B, T, dtype=DT)
    ids = torch.arange(B).view(B, 1).expand(B, T)   # row b -> sample b
    w = _weights(ids, mask, min_tokens=2)
    one = _fused("grpo_sample", h, W, sel, mask, adv, sample_weight=w, chunk_size=B)   # chunks=1
    many = _fused("grpo_sample", h, W, sel, mask, adv, sample_weight=w, chunk_size=1)  # chunks=4
    ref = _ref_sum(-adv, mask, ids, 2)         # ratio=1 -> per_token_loss = -adv
    # kernel accumulates loss_acc in fp32 across chunks -> fp32-epsilon tolerance.
    assert torch.allclose(one.double(), many.double(), rtol=1e-5, atol=1e-6)
    assert torch.allclose(many.double(), ref.double(), rtol=1e-5, atol=1e-6)


def test_grpo_sample_gradient_flows():
    torch.manual_seed(7)
    B, T, D, V = 1, 9, 8, 16
    h = torch.randn(B, T, D, dtype=DT, requires_grad=True)
    W = torch.randn(V, D, dtype=DT)
    sel = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=DT)
    adv = torch.randn(B, T, dtype=DT)
    ids = _sample_ids([0, 4, 9], T)
    out = _fused("grpo_sample", h, W, sel, mask, adv, sample_weight=_weights(ids, mask, 2))
    g = torch.autograd.grad(out, h)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0
