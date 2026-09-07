#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_loss.py  (20260609)  — visualization_analysis.py から loss 描画を分離
=========================================================================

loss_details.csv（step, original_loss, reg_value, reg_weighted, total_loss, ...）から
loss 曲線を描く。モデルやデータは不要（CSV のみ）。

  - loss_curves.png         : twinx（loss 軸 + reg 軸）
  - loss_curves_simple.png  : 単一軸

単体実行: python plot_loss.py --loss_path <loss_details.csv> --output_dir <dir>
run_all_viz.py からは {viz}/loss を --output_dir として呼ばれる。
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(loss_csv_path, output_dir):
    if not loss_csv_path or not os.path.exists(loss_csv_path):
        print(f"[loss] loss csv not found: {loss_csv_path}")
        return
    df = pd.read_csv(loss_csv_path)
    # twinx 版
    fig, ax1 = plt.subplots(figsize=(12, 8))
    if "original_loss" in df:
        ax1.plot(df["step"], df["original_loss"], color="blue", lw=2, alpha=0.8, label="Reconst Error")
    if "total_loss" in df:
        ax1.plot(df["step"], df["total_loss"], color="red", lw=2, alpha=0.8, label="Total Loss")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.grid(True, alpha=0.3)
    if "reg_weighted" in df:
        ax2 = ax1.twinx()
        ax2.plot(df["step"], df["reg_weighted"], color="green", lw=1.5, alpha=0.7, label="reg (weighted)")
        ax2.set_ylabel("regularization")
    ax1.legend(loc="upper right"); plt.title("Loss curves")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=200); plt.close()

    # simple 版
    plt.figure(figsize=(12, 7))
    for col, color in (("original_loss", "blue"), ("total_loss", "red"), ("reg_weighted", "green")):
        if col in df:
            plt.plot(df["step"], df[col], color=color, lw=2, alpha=0.8, label=col)
    plt.xlabel("step"); plt.ylabel("value"); plt.legend(); plt.grid(True, alpha=0.3); plt.title("Loss (simple)")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "loss_curves_simple.png"), dpi=200); plt.close()
    print("[loss] loss curves saved")


def main():
    a = create_argparser().parse_args()
    out = a.output_dir if a.output_dir else os.getcwd()
    os.makedirs(out, exist_ok=True)
    plot_loss_curves(a.loss_path, out)
    print(f"[loss] done. Output directory: {out}")


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--loss_path", default="", help="loss_details.csv")
    p.add_argument("--output_dir", default="")
    return p


if __name__ == "__main__":
    main()
