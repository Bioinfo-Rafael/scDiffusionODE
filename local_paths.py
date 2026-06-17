"""
local_paths.py
==============

data_dir / edge_tsv_path のローカル fallback。

方針: 既定はリモートパス（/home/suzuki/Projects/scDiffusion/...）のまま完全維持。
設定パスが**存在すればそのまま使う**（= リモート、no-op）。**存在しなければ**このリポジトリ
直下へ re-root して探す（= ローカルテスト）。ホスト判定はせずパス存在で分岐するため、
リモートでは挙動が一切変わらない。
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REMOTE_ROOT = "/home/suzuki/Projects/scDiffusion"


def resolve_path(p):
    """存在するパスはそのまま返す。リモート形式 or 相対で存在しない場合のみ repo 直下へ解決。"""
    if not p or os.path.exists(p):
        return p
    s = str(p)
    # リモート絶対パス -> repo 直下に re-root
    if s.startswith(REMOTE_ROOT):
        cand = os.path.join(REPO_ROOT, os.path.relpath(s, REMOTE_ROOT))
        if os.path.exists(cand):
            return cand
    # 相対パス -> repo 基準でも探す
    if not os.path.isabs(s):
        cand = os.path.join(REPO_ROOT, s)
        if os.path.exists(cand):
            return cand
    # 解決できなければ元のまま（従来どおり呼び出し側で失敗させる）
    return p
