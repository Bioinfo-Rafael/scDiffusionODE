#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_paths.py  (20260609)
========================

出力ディレクトリ構造を 3 dir 共通で扱う小さなヘルパ。

    {work_root}/runs/{model_name}/{YYYYMMDD_HHMMSS}/{train,sample,viz}/...

- model_name は記述的（番号でない）。日付 dir は **時刻 HHMMSS まで**含むので実行ごとにユニーク。
- 同じモデルを同日に複数回流しても {YYYYMMDD_HHMMSS}/ で完全分離（内部 {ts}_train も残る）。
- pipeline / matrix は base を一度採番して各 stage に渡す。
- 単体実行の sample/viz は --model_path から base を逆算する（infer_run_base）。

Hybrid5x3 内に置き、他 2 dir からは
    sys.path.insert(0, os.path.join(HERE, "..", "20260609_Hybrid5x3"))
    import run_paths
で読む（既存の VIZ パス参照と同じ手）。依存は標準ライブラリのみ。
"""

import os
from datetime import datetime


def run_base(work_root, model_name, when=None, create=True):
    """{work_root}/runs/{model_name}/{YYYYMMDD_HHMMSS} を返す（実行ごとにユニーク）。

    model 名 dir を runs/ 配下に置くことで、work dir 直下が 15 モデル分で散らからない。
    日付 dir は時刻 HHMMSS まで含むので、同日の再実行も別 dir に完全分離される。
    when を渡すと採番時刻を固定できる（matrix で全実験を 1 つの batch 時刻に揃える用途）。
    create=False のときはパス文字列のみ返す（dry-run でディレクトリを作らない用途）。
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(work_root, "runs", model_name, stamp)
    if create:
        os.makedirs(base, exist_ok=True)
    return base


def stage_dir(base, stage):
    """{base}/{stage} を作成して返す（stage ∈ train/sample/viz）。"""
    d = os.path.join(base, stage)
    os.makedirs(d, exist_ok=True)
    return d


def infer_run_base(model_path):
    """model_path から {model_name}/{date} の base を逆算する（単体 sample/viz 用）。

    例: .../{base}/train/{ts}_train/checkpoints/{name}/ema_*.pt → {base}
    最後の 'train' セグメントの親を返す。見つからなければ dirname フォールバック。
    """
    if not model_path:
        return ""
    parts = os.path.abspath(model_path).split(os.sep)
    if "train" in parts:
        i = len(parts) - 1 - parts[::-1].index("train")  # 最後の 'train' の位置
        return os.sep.join(parts[:i])
    return os.path.dirname(os.path.abspath(model_path))
