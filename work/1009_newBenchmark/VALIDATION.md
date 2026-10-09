# 検証記録

実行日: **2026-10-09 JST**。ローカルmacOS、既存 `scdiffusion` 環境（Python 3.9 / Torch 2.5.1 / Scanpy 1.9.3 / scVelo 0.2.5）、CPU。環境へのinstall/updateはなし。

## 結果

- **8 tests成功、7.390秒**。ログ: `runs/logs/tests.log`。
- 80細胞×64遺伝子→32 HVG。Erythroid60細胞。学習用Xは元の線形値と完全一致し、cell/gene IDs・元列対応・入力ファイルhashを保持。
- テスト内だけhidden幅を `[16,12,8,8]` に縮小。実際の既存Cell_Unet、START_X MSE、AdamW、EMAを使用。production設定 1,024 HVG / `[2000,1000,500,500]` / 30,000 updatesの取得をassert。
- 2 updatesのcheckpointから12 updatesへresume。epoch境界をまたぎ、連続12 updatesとraw/EMA全tensorがbit単位で一致。CPUでの確認であり異なるGPU/OS間の再現性は保証しない。
- 1,000段のancestral samplingを実行し4細胞を生成。有限値・gene順・偽celltypeなし・同条件成果物の再利用を確認。
- 生のx_startと再構成変位をh5adへexport。`velocity`層・`benchmark_velocity` metadataが存在しないこと、変位が予測−Xであることを確認。
- samples、all、erythroidの3 scopeでUMAP、全体/Erythroidそれぞれのstream/arrow/gridを実際に描画。Erythroidは全体UMAPの単なる切出しではないことをassert。
- syntheticのstream図・実/生成比較図を画像として開き、凡例と `NOT RNA velocity` 表記を確認。未学習に近いsynthetic結果であり、生物学的品質評価ではない。
- gene/cell ID順序・ID改変・重複ID・二重前処理・Xとsplicedの不一致を拒否。
- CellUNetに対してVeloEV subprocessを起動しないことをmockでassert。通常CLIもexit **2**、`not_applicable`、4指標の値=null、成功スコアCSVなし。
- 6 CLIの `--help` はすべてexit 0。Python構文と `bash -n run_all.sh` 成功。
- 作業前に記録した既存 **1,011ファイル** のSHA256を照合し、変更 **0件**。元から存在したユーザー編集もそのまま。`runs/logs/integrity.json` に保存。

最終synthetic成果物: `runs/logs/synthetic_ts76dkzy/run/`。
Scanpy/Matplotlib/scVeloの旧APIに由来するdeprecation warningとOpenMPのnoticeはログに保持。テスト失敗なし。

## 未実行・対象外

`data_preparation/20261007/data/MouseGastrulation.h5ad` はこの環境に存在しない。実データ準備、production幅での本学習、実3,000細胞生成、全89,267/Erythroid9,815細胞可視化、CUDA実行、全体のピークRAM/実行時間は未検証。本学習を自動開始していない。

benchmark referenceの再作成・専用venvへの変更は行っていない。CellUNet Stage 1から生物学的ds/dtを定義できないため、VeloEVのCBDir/ICVCoh/CTO/TSC実計算は意図的に実行しない。結果は `runs/metrics/benchmark_status.json` と `applicability.csv` に `not_applicable` として記録した。
