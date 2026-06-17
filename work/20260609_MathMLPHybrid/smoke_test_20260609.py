#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for math-MLP hybrid fields (20260609). 学習マシン上で実行する。

torch のみ必要（h5ad / edge tsv 不要：mask はランダム生成で代用）。

  python smoke_test_20260609.py

検証項目:
  1. import
  2. forward shape  ((B,d),(B,1)) -> (B,d)
  3. MathML_Hybrid forward -> (B,d)、train_util hook の duck-type
     (model.ode_model.soft / off_mask_penalty(norm))
  4. off_mask_penalty が scalar、l1/l2、mask 有無で値が変わる
  5. reg が backprop して field の grad が立つ
  6. compute_W -> (n_vis,d,d)
  7. checkpoint save/load (strict=False) で出力一致
  8. forward 中に batched W (B*d*d) を確保していないこと（CUDA時のみ概算）
"""
import sys, copy
sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')
import torch

from ODE.ode_20260609_mathmlp import (
    LowRankField, LinCombField, MatSumField, LoRAField,
    MathML_Hybrid, build_math_field, load_hybrid_state_dict,
)
from guided_diffusion.cell_model import Cell_Unet

B, d, r, K, T = 8, 64, 4, 4, 1000
dev = "cuda" if torch.cuda.is_available() else "cpu"
mask = (torch.rand(d, d) > 0.7).float()       # ランダム mask で代用


def make(model_type, use_mask=True):
    m = mask if use_mask else None
    if model_type == "lowrank":
        return LowRankField(d, rank=r, mask=m, hidden=64, time_dim=32)
    if model_type == "lincomb":
        return LinCombField(d, K=K, mask=m, hidden=64, time_dim=32)
    if model_type == "matsum":
        return MatSumField(d, K=K, mask=m, hidden=64, time_dim=32)
    if model_type == "lora":
        return LoRAField(d, K=K, rank=r, mask=m, hidden=64, time_dim=32)


def run(model_type):
    print(f"\n=== {model_type} ===")
    x = torch.randn(B, d, device=dev)
    t = torch.randint(0, T, (B, 1), device=dev)

    field = make(model_type).to(dev)
    field.train()

    # 2. field forward shape
    out = field(x, t)
    assert out.shape == (B, d), out.shape
    print("  forward shape OK", tuple(out.shape))

    # 2b. GeneODE 風 decay: gamma は (d,) で K 非依存、出力は符号付き
    assert tuple(field.gamma.shape) == (d,), field.gamma.shape
    assert field.use_decay
    print(f"  gamma shape OK {tuple(field.gamma.shape)} (K-independent); "
          f"out min={out.min().item():.3f} max={out.max().item():.3f} "
          f"has_negative={(out < 0).any().item()}")

    # 2c. t を省略 / NaN でも落ちない（t=0 扱い、GeneODE 互換）
    o_none = field(x)
    o_nan = field(x, torch.full((B, 1), float("nan"), device=dev))
    assert o_none.shape == (B, d) and o_nan.shape == (B, d)
    print("  t=None / NaN forward OK (treated as t=0)")

    # 3. hybrid + duck-type hook
    ml = Cell_Unet(input_dim=d).to(dev)
    hybrid = MathML_Hybrid(field, ml, timesteps=T).to(dev)
    hybrid.train()
    hout = hybrid(x, t.float())
    assert hout.shape == (B, d)
    assert hybrid.ode_model is field and getattr(field, "soft", False)
    print("  hybrid forward + duck-type ode_model OK")

    # 4. off_mask_penalty scalar + l1/l2 + mask on/off
    p1 = field.off_mask_penalty("l1"); p2 = field.off_mask_penalty("l2")
    assert p1.dim() == 0 and p2.dim() == 0
    field_nomask = make(model_type, use_mask=False).to(dev); field_nomask.train()
    field_nomask(x, t)
    pn = field_nomask.off_mask_penalty("l1")
    print(f"  penalty l1={p1.item():.5f} l2={p2.item():.5f} nomask_l1={pn.item():.5f}")

    # 4b. lowrank の cache-gating / 防御: enable_offmask_cache=False や eval で 0
    if model_type == "lowrank":
        field.enable_offmask_cache = False
        field(x, t)
        assert field._cached_W_sub is None
        assert field.off_mask_penalty("l1").item() == 0.0
        field.enable_offmask_cache = True
        field.eval(); field(x, t)
        assert field._cached_W_sub is None  # eval ではキャッシュしない（stale 防止）
        assert field.off_mask_penalty("l1").item() == 0.0
        field.train()
        print("  lowrank cache-gating + None-guard OK")

    # 5. backprop into field params
    hybrid.zero_grad()
    loss = hout.pow(2).mean() + 0.1 * field.off_mask_penalty("l1")
    loss.backward()
    g = [p.grad.abs().sum().item() for p in field.parameters() if p.grad is not None]
    assert any(v > 0 for v in g), "no grad in field"
    print(f"  backprop OK ({len(g)} grad tensors)")

    # 6. compute_W + proxy flag（lincomb のみ W_IS_EXACT=False）
    W = field.compute_W(x[:2], int(t[0].item()))
    assert W.shape == (2, d, d), W.shape
    expect_exact = (model_type != "lincomb")
    assert field.W_IS_EXACT == expect_exact, (model_type, field.W_IS_EXACT)
    print(f"  compute_W shape OK {tuple(W.shape)}  W_IS_EXACT={field.W_IS_EXACT}")

    # 7. checkpoint save/load via safe loader (strict=True, core key 検証)
    sd = copy.deepcopy(hybrid.state_dict())
    field2 = make(model_type).to(dev); ml2 = Cell_Unet(input_dim=d).to(dev)
    hybrid2 = MathML_Hybrid(field2, ml2, timesteps=T).to(dev)
    load_hybrid_state_dict(hybrid2, sd, strict=True, log=lambda *a: None)
    hybrid.eval(); hybrid2.eval()
    with torch.no_grad():
        o1 = hybrid(x, t.float()); o2 = hybrid2(x, t.float())
    assert torch.allclose(o1, o2, atol=1e-5), "checkpoint roundtrip mismatch"
    print("  ckpt roundtrip OK (safe loader, strict=True)")

    # 7b. 間違った model_type の checkpoint は strict=True で止まること
    wrong_type = "matsum" if model_type != "matsum" else "lora"
    fw = make(wrong_type).to(dev); mw = Cell_Unet(input_dim=d).to(dev)
    hybrid_wrong = MathML_Hybrid(fw, mw, timesteps=T).to(dev)
    raised = False
    try:
        load_hybrid_state_dict(hybrid_wrong, copy.deepcopy(sd), strict=True, log=lambda *a: None)
    except RuntimeError:
        raised = True
    assert raised, f"wrong checkpoint ({model_type}->{wrong_type}) should raise under strict=True"
    print("  wrong-checkpoint guard OK (strict=True raises)")


def main():
    print(f"device={dev}, B={B} d={d} r={r} K={K}")
    for mt in ["lowrank", "lincomb", "matsum", "lora"]:
        run(mt)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(B, d, device=dev); t = torch.randint(0, T, (B, 1), device=dev)
        f = make("lowrank").to(dev); f.train(); f(x, t)
        peak = torch.cuda.max_memory_allocated() / 1e6
        batched_W_MB = B * d * d * 4 / 1e6
        print(f"\n[mem] lowrank forward peak={peak:.1f}MB ; batched W would be {batched_W_MB:.1f}MB")
    print("\nALL SMOKE TESTS PASSED ✅")


if __name__ == "__main__":
    main()
