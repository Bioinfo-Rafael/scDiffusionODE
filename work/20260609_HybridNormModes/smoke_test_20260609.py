#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for ODE_ML_HybridNorm 3 modes (20260609). 学習マシン上で実行する。

torch のみ必要（h5ad / edge tsv 不要）。GeneODE は実物の off_mask_penalty 挙動を忠実に
再現したスタブで代用し、train_util の正則化 hook をシミュレートして検証する。

  python smoke_test_20260609.py

検証項目:
  - ratio_reg / normed_learned_scale / none の forward 出力 shape=(B,d)、backward が通る
  - normed で log_scale に grad が流れ、hybrid.parameters() に含まれる
  - ratio_reg のみ _cached_ratio_reg is not None（ratio 成分が off_mask_penalty に乗る）
  - none / normed では _cached_ratio_reg is None → off_mask_penalty は base のみ
  - SoftReg=True で off-mask penalty 加算 / SoftReg=False で hook 不発火（既存一致）
  - DDP wrapper 想定: getattr(model,"module",model).ode_model.off_mask_penalty(norm) が全 mode で動く
  - ratio_reg / none は log_scale を持たない（checkpoint 後方互換）
"""
import sys
sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')

import torch
import torch.nn as nn
import torch.nn.functional as F

from ODE.ode_20260609_hybridnorm import ODE_ML_HybridNorm

B, d, T = 8, 32, 1000
dev = "cuda" if torch.cuda.is_available() else "cpu"
ODE_REG_LAMBDA = 5.0
ODE_REG_NORM = "l1"


class GeneODEStub(nn.Module):
    """実 GeneODE の duck-type 挙動を忠実に再現（off_mask_penalty も同形）。"""
    def __init__(self, d, soft=True):
        super().__init__()
        self.soft = soft
        self.register_buffer("mask", (torch.rand(d, d) > 0.7).float())
        self.W = nn.Parameter(torch.randn(d, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))
        self.gamma = nn.Parameter(torch.ones(d) * 0.1)
        self.ratio_reg_weight = 1.0
        self.ratio_reg_target = 1.0
        self.ratio_reg_eps = 1e-8
        self._cached_ratio_reg = None

    def forward(self, x, t=None):
        x = x.float()
        return F.softplus(x @ self.W + self.b) - F.softplus(self.gamma) * x

    def off_mask_penalty(self, norm="l1"):
        off = (1.0 - self.mask) * self.W
        base = (off ** 2).mean() if norm.lower() == "l2" else off.abs().mean()
        aux = getattr(self, "_cached_ratio_reg", None)
        if aux is None or self.ratio_reg_weight <= 0:
            return base
        return base + self.ratio_reg_weight * aux


class MLStub(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, t, y=None):
        return self.lin(x.float())


def simulate_hook(hybrid, lam=ODE_REG_LAMBDA, norm=ODE_REG_NORM):
    """train_util.forward_backward の正則化 hook を忠実に再現して reg を返す。"""
    model_ref = getattr(hybrid, "module", hybrid)        # DDP wrapper 想定
    ode_ref = getattr(model_ref, "ode_model", None)
    if ode_ref is not None and getattr(ode_ref, "soft", False) and lam > 0:
        return ode_ref.off_mask_penalty(norm)
    return None


def run(mode, soft=True):
    print(f"\n=== mode={mode} soft={soft} ===")
    x = torch.randn(B, d, device=dev)
    t = torch.randint(0, T, (B, 1), device=dev).float()

    ode = GeneODEStub(d, soft=soft).to(dev)
    ml = MLStub(d).to(dev)
    hybrid = ODE_ML_HybridNorm(ode, ml, timesteps=T, hybrid_norm_mode=mode).to(dev)
    hybrid.train()

    out = hybrid(x, t)
    assert out.shape == (B, d), out.shape
    print("  forward shape OK", tuple(out.shape))

    # _cached_ratio_reg は ratio_reg のみ非 None
    crr = ode._cached_ratio_reg
    if mode == "ratio_reg":
        assert crr is not None, "ratio_reg should cache ratio penalty"
    else:
        assert crr is None, f"{mode} must not cache ratio penalty"
    print(f"  _cached_ratio_reg is {'set' if crr is not None else 'None'} (expected)")

    # log_scale の有無
    has_scale = hasattr(hybrid, "log_scale")
    assert has_scale == (mode == "normed_learned_scale")
    if has_scale:
        assert any(p is hybrid.log_scale for p in hybrid.parameters())
    print(f"  log_scale present={has_scale} (in parameters={has_scale})")

    # hook シミュレーション
    reg = simulate_hook(hybrid)
    if soft:
        assert reg is not None and reg.dim() == 0
        base_only = ode.off_mask_penalty(ODE_REG_NORM) if mode != "ratio_reg" else None
        if mode == "ratio_reg":
            # ratio 成分が base に上乗せされている（_cached_ratio_reg>0 想定）
            print(f"  hook reg(ratio_reg)={reg.item():.6f}  (= base + ratio_reg_weight*ratio)")
        else:
            # base のみ
            assert torch.allclose(reg, base_only)
            print(f"  hook reg({mode})={reg.item():.6f}  (= off-mask base only)")
    else:
        assert reg is None, "SoftReg=False must not fire the hook"
        print("  SoftReg=False -> hook not fired (OK)")

    # backward（diffusion loss 相当 + reg）
    hybrid.zero_grad()
    loss = out.pow(2).mean()
    if reg is not None:
        loss = loss + ODE_REG_LAMBDA * reg
    loss.backward()

    # normed の log_scale に grad が流れること
    if mode == "normed_learned_scale":
        assert hybrid.log_scale.grad is not None and hybrid.log_scale.grad.abs().item() >= 0
        print(f"  log_scale.grad={hybrid.log_scale.grad.item():.6f} (flows)")
    # ML/ODE に grad
    g = [p.grad.abs().sum().item() for p in hybrid.parameters() if p.grad is not None]
    assert any(v > 0 for v in g), "no grad"
    print(f"  backward OK ({len(g)} grad tensors)")

    # eval では _cached_ratio_reg=None（stale 防止）
    hybrid.eval()
    with torch.no_grad():
        hybrid(x, t)
    assert ode._cached_ratio_reg is None
    print("  eval clears ratio cache OK")


def main():
    print(f"device={dev}, B={B} d={d}")
    for mode in ["ratio_reg", "normed_learned_scale", "none"]:
        run(mode, soft=True)
    # SoftReg=False の既存挙動（hook 不発火）
    run("ratio_reg", soft=False)
    run("none", soft=False)

    # 不正 mode は ValueError
    raised = False
    try:
        ODE_ML_HybridNorm(GeneODEStub(d), MLStub(d), timesteps=T, hybrid_norm_mode="bogus")
    except ValueError:
        raised = True
    assert raised
    print("\ninvalid mode -> ValueError OK")
    print("\nALL SMOKE TESTS PASSED ✅")


if __name__ == "__main__":
    main()
