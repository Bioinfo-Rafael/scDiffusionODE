#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_scale_model_forward.py  (20260609)
=======================================

scale_model mode 追加の軽量 forward test（実データ不要・ランダム tensor + dummy gene/edge）。

検証（§12）:
  1. Cell_Unet.forward(x,t) -> (B,d)
  2. Cell_Unet.forward_with_features(x,t) -> ml_out (B,d), ml_features["ml_emb"] (B,hidden)
  3. forward(x,t) == forward_with_features(x,t)[0]（値一致）
  4. hybrid_norm_mode="ratio_reg" forward
  5. hybrid_norm_mode="none" forward
  6. hybrid_norm_mode="scale_model" + simple + ml_emb forward
  7. scale_model 出力 shape == (B,1)（かつ正値）
  8. hybrid 最終出力 (B,d)
  9. ode_input_source="none" が既存 ODE branch を壊さない
 10. ode_input_source="x_ml_emb" を未対応 branch で使うと明確な error
 (+) scale_model mode + scale_model_type="none" は build 時に明確な error
"""

import csv
import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in ("/home/suzuki/Projects/scDiffusion", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from guided_diffusion.cell_model import Cell_Unet           # noqa: E402
from ODE.ode_20260609_hybrid5x3 import build_denoiser       # noqa: E402


def _make_edge_tsv(genes, path):
    edges = [(genes[0], genes[1]), (genes[1], genes[2]),
             (genes[3], genes[4]), (genes[2], genes[5])]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["from", "to"])
        for a, b in edges:
            w.writerow([a, b])


def main():
    torch.manual_seed(0)
    B, d, T = 4, 20, 1000
    genes = [f"g{i}" for i in range(d)]
    tmp = tempfile.mkdtemp()
    tsv = os.path.join(tmp, "edges.tsv")
    _make_edge_tsv(genes, tsv)
    x = torch.randn(B, d)
    t = torch.randint(1, T, (B,)).long()

    results = []

    def check(name, cond):
        cond = bool(cond)
        results.append((name, cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    # --- 1-3: Cell_Unet forward / forward_with_features ---
    ml = Cell_Unet(input_dim=d).eval()
    with torch.no_grad():
        o1 = ml(x, t)
        o2, feats = ml.forward_with_features(x, t)
    hdim = ml.hidden_num[-1]
    check("1  Cell_Unet.forward -> (B,d)", tuple(o1.shape) == (B, d))
    check("2a forward_with_features ml_out -> (B,d)", tuple(o2.shape) == (B, d))
    check(f"2b ml_features['ml_emb'] -> (B,{hdim})", tuple(feats["ml_emb"].shape) == (B, hdim))
    check("3  forward == forward_with_features[0]", torch.allclose(o1, o2, atol=1e-6))

    # --- 4-5: ratio_reg / none forward（既存挙動）---
    for i, mode in (("4", "ratio_reg"), ("5", "none")):
        m = build_denoiser("geneode", genes, tsv, timesteps=T,
                           hybrid_norm_mode=mode, device="cpu").eval()
        with torch.no_grad():
            out = m(x, t)
        check(f"{i}  hybrid_norm_mode='{mode}' forward -> (B,d)", tuple(out.shape) == (B, d))

    # --- 6-9: scale_model + simple + ml_emb ---
    m = build_denoiser("geneode", genes, tsv, timesteps=T, hybrid_norm_mode="scale_model",
                       scale_model_type="simple", scale_input_source="ml_emb",
                       ode_input_source="none", scale_hidden=32, device="cpu").eval()
    with torch.no_grad():
        out = m(x, t)
        _, feats2 = m.ml_model.forward_with_features(x, t)
        sc = m.scale_model(feats2["ml_emb"], t)
    check("6  scale_model mode forward runs", out is not None)
    check("7a scale_model output -> (B,1)", tuple(sc.shape) == (B, 1))
    check("7b scale positive", bool((sc > 0).all()))
    check("8  hybrid final output -> (B,d)", tuple(out.shape) == (B, d))
    check("9  ode_input_source='none' OK (no crash)", tuple(out.shape) == (B, d))

    # --- 10: x_ml_emb on unsupported branch -> RuntimeError ---
    m2 = build_denoiser("geneode", genes, tsv, timesteps=T, hybrid_norm_mode="scale_model",
                        scale_model_type="simple", scale_input_source="ml_emb",
                        ode_input_source="x_ml_emb", scale_hidden=32, device="cpu").eval()
    raised = False
    try:
        with torch.no_grad():
            m2(x, t)
    except RuntimeError as e:
        raised = True
        print(f"   expected RuntimeError: {str(e)[:90]}")
    check("10 ode_input_source='x_ml_emb' (unsupported) -> RuntimeError", raised)

    # --- (+) scale_model mode + type 'none' -> build error ---
    raised2 = False
    try:
        build_denoiser("geneode", genes, tsv, timesteps=T, hybrid_norm_mode="scale_model",
                       scale_model_type="none", device="cpu")
    except ValueError as e:
        raised2 = True
        print(f"   expected ValueError: {str(e)[:90]}")
    check("+  scale_model mode + scale_model_type='none' -> error", raised2)

    n_pass = sum(1 for _, c in results if c)
    print(f"\n=== {n_pass}/{len(results)} PASS ===")
    if n_pass != len(results):
        print("SOME_FAIL")
        sys.exit(1)
    print("ALL_FORWARD_TESTS_PASS")


if __name__ == "__main__":
    main()
