"""Three arms for W1.2, sharing one encoder and one output contract.

Every arm: forward(x[B,L,C]) -> (y_norm[B,H,C], mu[B,1,C], sd[B,1,C]).
y_norm is in instance-normalised space; train.py computes the loss there and
denormalises for reporting, which is the standard PatchTST/RevIN protocol.

  plain   channel-independent encoder + linear head. No cross-channel information.
  concat  [z_c ; mu_c ; log sd_c] per channel -> token -> transformer over channels.
          This is the "trivial baseline": statistics enter by plain concatenation and
          channels mix freely. If this matches `sheaf`, the readout layer is unnecessary.
  sheaf   frozen-able CI encoder + additive per-source readout:
              y_i = head(h_i) + sum_j  gate_ij * m_j,
          gate_ij = sigmoid(<q_i, k_j>/sqrt(d) + delta*rho_ij + eta*(mu_i - mu_j))
          Additive over sources => deleting source j exactly removes its term.
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
        self.head = nn.Linear(self.d, H)

    def forward(self, x):
        h, mu, sd = self.embed(x)                       # (B,C,d)
        y = self.head(h).permute(0, 2, 1)               # (B,H,C)
        return y, mu, sd


class ConcatMix(nn.Module):
    """The trivial baseline: statistics as extra channels, channels mixed by attention."""

    def __init__(self, L, H, d=128, n_layer=2, n_head=8, dropout=0.1, C=64):
        super().__init__()
        self.norm = InstanceNorm()
        # one embedding per channel, sized to C: a fixed 64-row table breaks the
        # moment a dataset has more channels than that (Electricity=321, Traffic=862)
        self.chan = nn.Parameter(torch.zeros(1, max(C, 64), d))
        nn.init.trunc_normal_(self.chan, std=0.02)
        self.proj = nn.Linear(L + 2, d)
        layer = nn.TransformerEncoderLayer(
            d, n_head, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.head = nn.Linear(d, H)

    def forward(self, x):                       # (B, L, C)
        z, mu, sd = self.norm(x)
        s = torch.cat([z, mu.expand(-1, 1, -1), sd.log()], dim=1)   # (B, L+2, C)
        tok = self.proj(s.permute(0, 2, 1))                          # (B, C, d)
        tok = tok + self.chan[:, : tok.shape[1]]
        tok = self.enc(tok)
        y = self.head(tok).permute(0, 2, 1)                          # (B, H, C)
        return y, mu, sd


class SheafReadout(Base):
    """Additive per-(target, source) readout with the rho edge prior.

    Two scale details matter and are easy to get wrong:
      * the aggregate is a MEAN over sources, not a sum -- with C=21 a plain sum is
        ~10x the scale of one message and swamps the self branch;
      * the message branch is scaled by `msg_scale` initialised to 0, so the model
        STARTS EXACTLY AS `plain` and has to earn any cross-channel gain. This is
        the same ReZero convention PAC-Former uses for its SCA (`sca_gamma_init=0`).
    Without the second one the readout starts far from `plain` and is structurally
    guaranteed to underperform it.
    """

    def __init__(self, H, enc_kw, rho=None, dropout=0.1):
        super().__init__(enc_kw)
        d = self.d
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        # target-side gate sigma(U h_i) and source-side message V h_j, per the
        # report's c_ij = a_ij * P[ V h_j * sigma(U h_i) ].  Because sigma(U h_i)
        # does not depend on j it factors OUT of the sum, so we never materialise
        # a C x C x d tensor -- the whole thing stays O(C^2) in the gate matrix.
        self.u = nn.Linear(d, d)          # target-side, j-independent
        self.v = nn.Linear(d, d, bias=False)   # source-side, linear in h_j
        self.out_self = nn.Linear(d, H)
        # bias MUST be off: with a bias, out_msg(sum_j g_ij m_j) and
        # sum_j out_msg(g_ij m_j) differ by (C-1)*bias, so per-source deletion would
        # no longer be exact. Any bias belongs in out_self.
        self.out_msg = nn.Linear(d, H, bias=False)
        self.msg_scale = nn.Parameter(torch.zeros(1))   # zero-init -> equivalent to plain
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.5)))  # gate temperature, tau=0.5
        self.drop = nn.Dropout(dropout)
        self.delta = nn.Parameter(torch.tensor(1.0))    # weight on the rho prior
        self.eta = nn.Parameter(torch.tensor(0.1))      # weight on the level gap
        if rho is None:
            rho = torch.zeros(1, 1)
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.float32))

    def forward(self, x):                       # (B, L, C)
        h, mu, sd = self.embed(x)                       # (B,C,d)
        C = h.shape[1]
        q, k = self.q(h), self.k(h)
        score = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(self.d)

        # prior + level gap; both are per-edge and enter the same logit
        rho = self.rho
        if rho.shape[0] != C:                           # channel subset changed
            rho = torch.zeros(C, C, device=h.device)
        score = score + self.delta * rho.unsqueeze(0)
        m = mu.squeeze(1)                               # (B, C)
        m = (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-5)
        gap = m.unsqueeze(2) - m.unsqueeze(1)           # (B,C,C), mu_i - mu_j
        score = score + self.eta * gap

        # temperature sharpening: sigmoid(s/tau) with tau<1 pushes gates apart.
        # It stays per-edge (no cross-source normalisation), so deleting source j
        # still removes exactly one term and nothing else.
        gate = torch.sigmoid(score * torch.exp(self.log_tau))          # (B,C,C)
        msg = self.v(h)                                 # (B,C,d)  linear in h_j
        # SUM over sources: exactly deletable (no /C, no softmax).
        agg = torch.einsum("bij,bjd->bid", self.drop(gate), msg)       # (B,C,d)
        # target-side gate; j-independent, so it commutes with the sum above
        u = torch.sigmoid(self.u(h)) * agg                             # (B,C,d)
        y = (self.out_self(h) + self.msg_scale * self.out_msg(u)).permute(0, 2, 1)
        return y, mu, sd


class LinearCI(nn.Module):
    """DLinear-style per-channel linear map: window -> horizon, no channel mixing.

    Reference point for the transformer arms. On ETTh1 a linear model reaches ~0.39,
    so if this arm also lands near 0.39 the data pipeline is sound and any gap is the
    transformer overfitting; if it does not, the pipeline is wrong.
    """

    def __init__(self, L: int, H: int):
        super().__init__()
        self.norm = InstanceNorm()
        self.fc = nn.Linear(L, H)

    def forward(self, x):                      # (B, L, C)
        z, mu, sd = self.norm(x)
        y = self.fc(z.permute(0, 2, 1)).permute(0, 2, 1)   # (B,H,C)
        return y, mu, sd


class LinearRawCI(nn.Module):
    """Same linear map as LinearCI but WITHOUT per-window normalisation.

    The published DLinear number (0.386 on ETTh1 96->96) is a plain linear map on
    training-standardised data with no RevIN. This arm isolates whether the per-window
    normalisation is what costs us ~20%.
    """

    def __init__(self, L: int, H: int):
        super().__init__()
        self.fc = nn.Linear(L, H)

    def forward(self, x):
        y = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        mu = torch.zeros_like(x[:, :1])
        sd = torch.ones_like(x[:, :1])
        return y, mu, sd


def build(arm: str, L: int, H: int, rho=None, d: int = 128, n_layer: int = 3,
          patch: int = 16, dropout: float = 0.1, C: int = 64):
    enc_kw = dict(L=L, patch=patch, d=d, n_layer=n_layer, n_head=8, dropout=dropout)
    if arm == "linear":
        return LinearCI(L, H)
    if arm == "linear_raw":
        return LinearRawCI(L, H)
    if arm == "plain":
        return PlainCI(H, enc_kw)
    if arm == "concat":
        # same d / depth knobs as the other arms so capacity can be matched
        return ConcatMix(L, H, d=d, n_layer=n_layer, n_head=8, dropout=dropout, C=C)
    if arm == "sheaf":
        return SheafReadout(H, enc_kw, rho=rho, dropout=dropout)
    raise ValueError(f"unknown arm {arm!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

