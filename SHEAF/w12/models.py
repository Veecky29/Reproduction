"""Arms, revision 2 (after the review of the negative result).

Every arm now shares the SAME channel-independent patch encoder; only the way
cross-channel information enters the output differs. That was the flaw in v1: the
old `concat` was a separate channel-mixing network, so a win could not be
attributed to "unfaithful mixing" rather than "a different architecture".

  plain        CI encoder -> linear head on [h_i ; mu_i ; log sd_i]
  concat       CI encoder -> full self-attention across channels -> head
  sheaf        CI encoder -> additive, exactly-deletable readout:
                   agg_i = sum_{j != i} g_ij * (V h_j)
                   u_i   = sigmoid(U h_i) * agg_i
                   y_i   = out_self([h_i ; mu_i ; log sd_i]) + alpha * out_msg(u_i)

Three fixes relative to v1, all from the same review:
  * window statistics now reach the SELF branch directly (before, they only reached
    the gate, while `concat` had them inside its token -- an unfair asymmetry);
  * the gate diagonal is masked to 0, so the self term cannot leak into the
    cross-channel sum and "self vs cross" is exactly separable;
  * alpha (msg_scale) starts at 0.1 instead of 0: a zero init makes the readout's
    gradient tiny, and val-based early stopping then parks the model on a plain-
    equivalent solution before the branch can open.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    """Per-window normalisation (RevIN-lite), stats returned so the model can denorm.

    Inputs are already globally standardised (global std = 1), so a window std far
    below 1 only means the channel is flat inside that window. The usual 1e-5 floor
    then amplifies a single spike by ~1e8 -- on Weather, whose `rain` / `raining`
    columns are zero in most windows, that alone blows the loss up. Flooring at 0.1
    caps the amplification at 10x and leaves normal windows (std ~1) untouched.
    """

    def __init__(self, floor: float = 0.1):
        super().__init__()
        self.floor = floor

    def forward(self, x):                      # (B, L, C)
        mu = x.mean(1, keepdim=True)           # (B, 1, C)
        sd = x.std(1, keepdim=True).clamp_min(self.floor)
        return (x - mu) / sd, mu, sd


def _stats(mu, sd):
    """(B,C,2) window-statistics block: mu_i and log sd_i per channel."""
    return torch.cat([mu.transpose(1, 2), sd.log().transpose(1, 2)], dim=-1)


class PatchEncoder(nn.Module):
    """Channel-independent patch transformer; returns (B, C, d)."""

    def __init__(self, L=96, patch=16, d=128, n_layer=3, n_head=8, dropout=0.1):
        super().__init__()
        assert L % patch == 0, (L, patch)
        self.n_patch = L // patch
        self.proj = nn.Conv1d(1, d, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, n_head, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.flat = nn.Linear(self.n_patch * d, d)

    def forward(self, z):                      # (B, L, C) -> (B, C, d)
        B, L, C = z.shape
        x = z.permute(0, 2, 1).reshape(B * C, 1, L)
        x = self.proj(x).transpose(1, 2) + self.pos      # (BC, n_patch, d)
        x = self.enc(x)
        return self.flat(x.reshape(B * C, -1)).reshape(B, C, -1)


class Base(nn.Module):
    def __init__(self, enc_kw):
        super().__init__()
        self.norm = InstanceNorm()
        self.enc = PatchEncoder(**enc_kw)
        self.d = enc_kw.get("d", 128)

    def embed(self, x):
        z, mu, sd = self.norm(x)
        return self.enc(z), mu, sd


class PlainCI(Base):
    def __init__(self, H, enc_kw):
        super().__init__(enc_kw)
        self.head = nn.Linear(self.d + 2, H)

    def forward(self, x):
        h, mu, sd = self.embed(x)
        y = self.head(torch.cat([h, _stats(mu, sd)], dim=-1)).permute(0, 2, 1)
        return y, mu, sd


class ChannelMixFair(Base):
    """The unfaithful control, holding the encoder fixed.

    Same CI patch encoder as `sheaf`; the only difference is that channels are
    mixed by full self-attention (with softmax, so deletion is not exact) instead
    of by an additive readout.
    """

    def __init__(self, H, enc_kw, C=64, n_mix=1, dropout=0.1):
        super().__init__(enc_kw)
        d = self.d
        self.chan = nn.Parameter(torch.zeros(1, max(C, 64), d))
        nn.init.trunc_normal_(self.chan, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, 8, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.mix = nn.TransformerEncoder(layer, n_mix, enable_nested_tensor=False)
        self.head = nn.Linear(d + 2, H)

    def forward(self, x):
        h, mu, sd = self.embed(x)
        C = h.shape[1]
        z = self.mix(h + self.chan[:, :C])
        y = self.head(torch.cat([z, _stats(mu, sd)], dim=-1)).permute(0, 2, 1)
        return y, mu, sd


class SheafReadout(Base):
    """Additive per-(target, source) readout with the rho edge prior."""

    ALPHA_INIT = 0.1

    def __init__(self, H, enc_kw, rho=None, dropout=0.1, C=64):
        super().__init__(enc_kw)
        d = self.d
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.u = nn.Linear(d, d)                       # target-side gate
        self.v = nn.Linear(d, d, bias=False)           # source-side message
        self.out_self = nn.Linear(d + 2, H)
        # bias off: out_msg(sum_j ...) must equal sum_j out_msg(...), otherwise
        # per-source deletion is off by (C-1)*bias.
        self.out_msg = nn.Linear(d, H, bias=False)
        self.msg_scale = nn.Parameter(torch.tensor(self.ALPHA_INIT))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("offdiag", 1.0 - torch.eye(C))
        if rho is None:
            rho = torch.zeros(1, 1)
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.float32))
        self.gate_stats = None

    def forward(self, x):
        h, mu, sd = self.embed(x)                       # (B,C,d)
        C = h.shape[1]
        q, k = self.q(h), self.k(h)
        score = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(self.d)
        rho = self.rho
        if rho.shape[0] != C:
            rho = torch.zeros(C, C, device=h.device)
        score = score + self.delta * rho.unsqueeze(0)
        m = mu.squeeze(1)
        m = (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-5)
        score = score + self.eta * (m.unsqueeze(2) - m.unsqueeze(1))

        gate = torch.sigmoid(score * torch.exp(self.log_tau)) * self.offdiag[:C, :C]
        msg = self.v(h)
        agg = torch.einsum("bij,bjd->bid", gate, msg)   # sum over j != i
        u = torch.sigmoid(self.u(h)) * agg
        y = (self.out_self(torch.cat([h, _stats(mu, sd)], dim=-1))
             + self.msg_scale * self.out_msg(u)).permute(0, 2, 1)
        self.gate_stats = (float(gate.mean()), float(gate.std()),
                           float(gate.max()))
        return y, mu, sd


class MixGated(Base):
    """CI encoder + switchable residual channel mixing.

        h'_i = h_i + beta_i * Mix(h)_i

    `Mix` is one softmax channel-attention head; `beta_i` is a learned per-channel
    scalar initialised at 0.1, so beta=0 recovers `plain` and beta=1 recovers dense
    mixing (`concat`). The model therefore spans both endpoints we already measured
    (9.4% apart on Electricity, 0 on Weather) and has to pick.
    """

    def __init__(self, H, enc_kw, C=64, beta_init=0.1, rank=16, dropout=0.1):
        super().__init__(enc_kw)
        d = self.d
        self.rank = rank
        self.p = nn.Linear(d, rank, bias=False)
        self.q = nn.Linear(d, rank, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.beta = nn.Parameter(torch.full((max(C, 1),), float(beta_init)))
        self.head = nn.Linear(d + 2, H)

    def forward(self, x):
        h, mu, sd = self.embed(x)
        C = h.shape[1]
        p, q = self.p(h), self.q(h)                    # (B,C,r)
        a = torch.einsum("bir,bjr->bij", p, q) / math.sqrt(self.rank)
        a = torch.softmax(a, dim=-1)                   # (B,C,C)
        m = self.o(torch.einsum("bij,bjd->bid", a, self.v(h)))
        beta = self.beta[:C].view(1, C, 1)
        h2 = h + beta * m
        self.beta_mean = float(self.beta[:C].mean())
        y = self.head(torch.cat([h2, _stats(mu, sd)], dim=-1)).permute(0, 2, 1)
        return y, mu, sd


def build(arm: str, L: int, H: int, rho=None, d: int = 128, n_layer: int = 3,
          patch: int = 16, dropout: float = 0.1, C: int = 64, beta_init: float = 0.1):
    enc_kw = dict(L=L, patch=patch, d=d, n_layer=n_layer, n_head=8, dropout=dropout)
    if arm == "plain":
        return PlainCI(H, enc_kw)
    if arm == "concat":
        return ChannelMixFair(H, enc_kw, C=C, n_mix=1, dropout=dropout)
    if arm == "mix":
        return MixGated(H, enc_kw, C=C, beta_init=beta_init, dropout=dropout)
    if arm == "sheaf":
        return SheafReadout(H, enc_kw, rho=rho, dropout=dropout, C=C)
    raise ValueError(f"unknown arm {arm!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

