#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for the 5×3 + baseline unified hybrid (20260609). 学習マシン上で実行する。

torch のみ必要。実 GeneODE / math fields を使うため edge_tsv は要るが、ダミー tsv を tmp に作る。

  python smoke_test_5x3.py

検証:
  - 5 ODE branch × 3 mode + 2 baseline の build_denoiser が通る
  - 各 forward 出力 shape=(B,d)、backward が通る
  - field 枝で t が伝播し時刻依存が残る（t=0 と t=T-1 で field 出力が変わる）
  - normed mode で log_scale に grad、hybrid.parameters() に含まれる
  - ratio_reg のみ ode_model._cached_ratio_reg is not None
  - train_util hook シミュレート: hybrid は off_mask_penalty 発火、plain は ode_model 無しで不発火
"""
import os
import sys
import tempfile

sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')

import torch

from ODE.ode_20260609_hybrid5x3 import build_denoiser, UnifiedODEMLHybrid

B, d, T = 8, 24, 1000
dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_dummy_tsv(gene_list):
    """gene_list 内で適当な制御エッジを持つ tf_target_edges.tsv を tmp に作る。"""
    fd, path = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w") as f:
        f.write("from\tto\n")
        for i in range(0, len(gene_list) - 1, 2):
            f.write(f"{gene_list[i]}\t{gene_list[i+1]}\n")
    return path


def simulate_hook(model, lam=5.0, norm="l1"):
    ode_ref = getattr(getattr(model, "module", model), "ode_model", None)
    if ode_ref is not None and getattr(ode_ref, "soft", False) and lam > 0:
        return ode_ref.off_mask_penalty(norm)
    return None


def run(ode_branch, mode, gene_list, tsv, soft=True, ode_reg_lambda=5.0):
    print(f"\n=== ode_branch={ode_branch} mode={mode} soft={soft} lam={ode_reg_lambda} ===")
    x = torch.randn(B, d, device=dev)
    t = torch.randint(0, T, (B, 1), device=dev).float()

    model = build_denoiser(
        ode_branch=ode_branch, gene_list=gene_list, edge_tsv_path=tsv, timesteps=T,
        hybrid_norm_mode=mode, rank=4, K=3, soft=soft, ode_reg_lambda=ode_reg_lambda, device=dev,
    )
    model.train()
    out = model(x, t)
    assert out.shape == (B, d), out.shape
    print(f"  forward OK {tuple(out.shape)}  is_hybrid={isinstance(model, UnifiedODEMLHybrid)}")

    # field 枝の時刻依存（t=0 vs t=T-1 で ODE 枝出力が変わること）
    if ode_branch in ("lowrank", "lincomb", "matsum", "lora"):
        with torch.no_grad():
            o0 = model.ode_model(x, torch.zeros(B, 1, device=dev))
            o1 = model.ode_model(x, torch.full((B, 1), float(T - 1), device=dev))
        assert not torch.allclose(o0, o1), "field ODE branch lost time-dependence!"
        print("  field time-dependence preserved (t=0 != t=T-1)")

    # ratio cache の出し分け
    if isinstance(model, UnifiedODEMLHybrid):
        crr = model.ode_model._cached_ratio_reg
        if mode == "ratio_reg":
            assert crr is not None
        else:
            assert crr is None
        print(f"  _cached_ratio_reg {'set' if crr is not None else 'None'} (expected)")
        # log_scale
        has_scale = hasattr(model, "log_scale")
        assert has_scale == (mode == "normed_learned_scale")
        if has_scale:
            assert any(p is model.log_scale for p in model.parameters())

    # hook シミュレート
    reg = simulate_hook(model, lam=ode_reg_lambda)
    if ode_branch == "plain":
        assert reg is None, "plain must not fire hook"
        print("  plain: hook not fired (OK)")
    elif soft and ode_reg_lambda > 0:
        assert reg is not None and reg.dim() == 0
        print(f"  hook reg={reg.item():.6f}")
    else:
        assert reg is None
        print("  soft/lam off -> hook not fired (OK)")

    # backward
    model.zero_grad()
    loss = out.pow(2).mean()
    if reg is not None:
        loss = loss + ode_reg_lambda * reg
    loss.backward()
    if isinstance(model, UnifiedODEMLHybrid) and mode == "normed_learned_scale":
        assert model.log_scale.grad is not None
        print(f"  log_scale.grad={model.log_scale.grad.item():.6f}")
    g = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert any(v > 0 for v in g), "no grad"
    print(f"  backward OK ({len(g)} grad tensors)")


def main():
    gene_list = [f"G{i}" for i in range(d)]
    tsv = make_dummy_tsv(gene_list)
    print(f"device={dev}, B={B} d={d}, dummy tsv={tsv}")

    branches = ["geneode", "lowrank", "lincomb", "matsum", "lora"]
    modes = ["ratio_reg", "normed_learned_scale", "none"]
    for br in branches:
        for m in modes:
            run(br, m, gene_list, tsv, soft=True, ode_reg_lambda=5.0)

    # baselines
    run("plain", "none", gene_list, tsv, soft=False, ode_reg_lambda=0.0)
    run("geneode", "none", gene_list, tsv, soft=False, ode_reg_lambda=0.0)

    os.remove(tsv)
    print("\nALL SMOKE TESTS PASSED ✅  (15 + 2 baseline = 17 configs)")


if __name__ == "__main__":
    main()
