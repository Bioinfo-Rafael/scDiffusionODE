# 検証記録 — 2026-10-27 JST

## 実施済み

- `python -m unittest discover -s work/20261027_matsum_lora_analysis/tests -v`: **13 tests passed**（1.976秒）。ログ: `outputs/unit_tests.log`。
- `python work/20261027_matsum_lora_analysis/run.py --smoke --device cpu --output work/20261027_matsum_lora_analysis/outputs/local_smoke_final`: **MatSum・LoRAとも全工程完了**。最終ログ時刻20:01:14 JST。
- 最終smokeの `manifest.json` は `status=complete`、`completed_models=[matsum,lora]`。記録された全解析Pythonファイルのhashが検証時点のファイルと一致。
- 人工48細胞・16遺伝子・3 celltype。元のMatSum/LoRAクラスを使用、学習は行わずランダム重みで検証。
- 出力: **322 PNG、132通常CSV、40 NPZ**（別途圧縮CSVとJSON）。各モデルで個別散布図28枚を確認。
- すべての8×8距離CSVの形状・対称性・対角ゼロを確認。
- Python11ファイル（本体9＋テスト2）の構文と空白を確認。
- 絶対値平均の注意マーク、8×8部分空間距離図、celltype別Cヒートマップを目視確認。ヒートマップはラベル重複を避けるレイアウトへ調整。

## 数値検証

| 検証 | MatSum | LoRA |
|---|---:|---:|
| 取得係数・基底からの再構成 vs 元forward：最大絶対誤差（smokeの先頭4細胞） | 4.5153e-7 | 1.1380e-6 |
| Wの全体分散 − celltype内分散 − celltype間分散：最大絶対残差 | 1.0408e-17 | 1.7347e-18 |

テストで別途確認した内容:

1. C/Wの解析的平均・stdが、全細胞行列を明示的に生成した参照計算と一致。
2. 符号付き平均と絶対値平均が異なる例、全ゼロ係数の正規化、平均ゼロ近傍のstd/mean除外。
3. 分散分解、incoming/outgoingの行列方向、自己ループ除外。
4. 対称化してから絶対値を取る順序、ラプラシアンの半正定値性・行和ゼロ・孤立点処理。
5. 行列の定数倍に対する通常/正規化ラプラシアンの挙動。
6. 部分空間比較の固有ベクトル符号反転・回転不変性、縮退検出。
7. ほぼ定数のヒストグラムでもbin幅が正であること。
8. 人工的に保存した**実checkpoint形式**で元D16/D20フィールドを復元し、EMA/rawの違いと元注釈のID結合を確認。
9. 遺伝子ID改変・非最終step・未レビューsource変更・非有限パラメータを拒否。
10. 全細胞バッチ集計による遺伝子スコアと成分除去効果が、直接計算した参照値と一致。

最初のsmokeで、係数付き成分のほぼ定数のstd/meanに対してヒストグラムbinが丸めで重なる問題を検出・修正。最終smokeではそのRuntimeWarningは発生していません。実行環境由来のOpenMP非推奨APIに関するInfoは出ています。

## 環境

macOS / CPU、既存の `scdiffusion` conda環境。NumPy 1.24.0、SciPy 1.10.0、pandas 2.0.0、PyTorch 2.5.1、matplotlib 3.7.0、scikit-learn 1.2.0、umap-learn 0.5.7、anndata 0.8.0、h5py 3.8.0。詳細はsmoke出力のmanifestにも記録。

## 未実施

接続先の学習済みD16/D20 checkpoint・実細胞89,267×1,024による `--check-only` と本解析、およびCUDAでの実行。このローカル環境には実checkpointがなく、操作可能なリモートトンネル接続も確認できなかったため。人工データの完走を、実細胞でのcelltype差や学習済みモデルの結果とは解釈しない。

## 起動・バックグラウンド実行の修正

ユーザー提供ログで、接続先のD16/D20の `latest.json` が想定の `model030000.pt` を指し、step=30000と記録されていることを確認。ログはMatplotlib import中の `KeyboardInterrupt` であり、checkpoint検証はまだ開始されていない。遅延の原因はこのログだけでは特定していない。

- `--help` は標準ライブラリのみで処理。起動ログをNumPy/PyTorch import前に表示。
- 描画・クラスタリングのimportを本解析へ移動。新しいプロセスでMatplotlib/scikit-learn/UMAP/解析モジュールのimportを禁止したまま、実形式の人工D16/D20 checkpointで `--check-only` が両方完了することを検証。
- `run_background.sh` はpreflightと本解析をnohupで順に実行。人工実行コマンドで、preflight成功時は2段階実行、終了コード7の失敗時は本解析を呼ばず終了コード7を保存することを確認。スペースを含むパスでも検証。
- **15 tests passed**（3.526秒）。ログ: `outputs/startup_tests_final.log`。shell構文とdiffの空白チェックも通過。
- 計算式は変更していない。遅延import後の本解析経路は `outputs/startup_fix_smoke/` の人工MatSumで全工程完了を確認（20:15:03 JST）。
