# CellRank環境分離の実装・検証（2026-10-29 JST）

親は既存`scdiffusion`（Python 3.9.23 / PyTorch 2.5.1）、子はCellRank専用Python 3.11.15とした。**親にはCellRankがなく、子にはPyTorchがない構成で、人工データの全解析と5枚の図生成が完了した。** 検証はローカルmacOS arm64。リモートへのインストール・ジョブ起動は行っていない。

## 変更範囲

- `CellRank/subprocess_bridge.py`と`worker.py`を追加。元発現・親が評価したVelocity・順序付きID・terminal指定・設定をNPZ/JSONで受け渡す。疎な遷移行列をCSR NPZで返す。
- 子は既存`cellrank_core.py`の公式APIを呼ぶ。同ファイル、D20 forward、学習コード、checkpoint、Julia solverは変更していない。
- `CellRankProjector.fit_scalar()`は`cellrank_python`または`CELLRANK_PYTHON`があれば子を使用する。未指定時は既存の同一プロセス経路を維持。
- 親でのCellRank/Scanpy強制importとCellRank version取得を除去。実モデルの読み込み、AD、補間・射影・集計・作図は親に残す。人工fixtureのUMAP作成だけは親の既存Scanpyを使う。
- `check_environment.py`が親のimport/JVPと子の実importを別々に確認する。インストール機能はない。
- バックグラウンドlauncherは独立OSセッションで子launcherを起動し、PID・ログ・終了コード・実行環境JSONを保存する。
- `requirements-cellrank.txt`は別Python 3.11環境にだけ入れる。親用`requirements.txt`の全体インストールは不要。子にTorch/CUDA/checkpointは不要。

## 検証結果

全体テストは **42 passed、1 skipped、2 performance warnings（64.19秒）**。skipはローカルにない実D20 checkpointの統合テストのみ。

追加した3テストで、競合terminalを含む公式の同一プロセス計算と子プロセス計算の一致、返却gene ID順序の不一致拒否、子の失敗時に親solverへ代替しないこと、失敗ログの保持、親でCellRank/Scanpy importを禁止してもfit・保存・再読込が成立することを検証した。

別途、既存Python 3.9/Torch 2.5.1で元の`LoRAField`を人工重みで構築し、JVPとfull jacrev、座標寄与とJacobian対角を比較した。平面上のD_S=-7.410262227058411、gene sum/decomposition残差0、重み不変を確認した。これは実D20 checkpointの確認ではない。

全工程smokeはN=200、D=10、概要40細胞、各クラス1代表、32 probes、seed 28。公式HodgeとCellRankを再計算した。結果は次の通り。

| 指標 | Hodge | CellRank |
|---|---:|---:|
| 成功数 | 40/40 | 34/40 |
| D_S平均 | 0.380469418562 | -2.309123182583 |
| 従来の単一環境smokeとの差の最大絶対値（成功細胞ごとのD_S） | 4.16e-10 | 2.85e-9 |

両構成で全200行のstatusは一致し、CellRankの6失敗は既存の`line_search_or_domain_failure`と一致した。したがって今回の人工例では、プロセス分離と親側依存バージョンの変更によるD_Sの差は上記の数値差に収まった。実データ全般で同じ誤差上限を保証するものではない。HodgeとCellRank間の数値差を、生物学的な因果効果とは扱わない。

バックグラウンド実行は`exit_code=0`、`completed_with_cell_failures`で完了し、PNGが5枚であることを確認した。検証出力は`/tmp/scprojection-split-final/job_20261029_113221_OqM9M6/analysis/`。生成物はGitへ追加せず、再現コードとこの記録を保存する。

子の`pip check`は`No broken requirements found.`。子で`find_spec('torch') is None`、preflightで`torch_imported=false`。親のpreflightは`cellrank_imported_in_parent=false`。作業前後の親環境のpackage名・version一覧183件は完全一致し、親にpip/conda installを実行していない。これはローカルの検証記録であり、リモート環境の状態を確認したという意味ではない。

## 再現と制約

[README冒頭](README.md)に、別環境の作成・`CELLRANK_PYTHON`の指定・起動コマンドを記載した。親は`conda activate scdiffusion`のまま。子の依存導入は`conda run -n scprojection-cellrank python -m pip ...`で対象を明示する。

```bash
# 検証で使用したローカル構成の例（各パスは環境に応じて指定）
CELLRANK_PYTHON=/tmp/scprojection-cellrank-only/bin/python \
  /Users/cls-lab/miniconda3/envs/scdiffusion/bin/python \
  work/20261028_Projection/scripts/check_environment.py

PROJECTION_TEST_CELLRANK_PYTHON=/tmp/scprojection-cellrank-only/bin/python \
JULIA=/tmp/julia-1.10.10/bin/julia JULIA_DEPOT_PATH=/tmp/scprojection-julia-depot \
  /tmp/scprojection-venv/bin/python -m pytest work/20261028_Projection/tests -q
```

Linux/GPU、実D20 checkpoint、全training細胞での時間・メモリは未検証。親の既存依存に不整合がある場合、今回の仕組みで自動修復はしない。子が別環境であることは、親で欠けているD20実行用依存を補うものではない。交換ファイルと失敗ログは本解析の出力配下に保持するため、全細胞X/V用の追加ディスク容量が必要。


2026-10-29追記：本報告の検証後、divergence entry pointのスカラー推定母集団もbenchmarkの5クラスに限定した。上記の過去の測定値は変更していない。母集団変更の回帰検証は`tests/test_scalar_population.py`を参照。
