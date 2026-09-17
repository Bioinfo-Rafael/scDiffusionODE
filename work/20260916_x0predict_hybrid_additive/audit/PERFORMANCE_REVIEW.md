# 2026-09-17 高速化調査と10kへの短縮

## 確認した状況

ユーザー提供のrun情報ではcentered_hill_otはcommit `5fa8f38`、2536/30000更新、
直近20.76秒/step。他のOT3条件は初回runの失敗記録であり、並列学習中ではありません。
reconstruction4条件は30kまで完了しています。リモートGPUの型番・現在の負荷・
実データに対する各区間の所要時間はこの環境から確認できていません。

## 実装から確定できる負荷

1. `5fa8f38`は200→400→600…と最初から再試行します。例えば550回で収束する問題は
   200+400+550=1150回のforward反復を行います。`fc4203a`以降は累計550回です。
   最大16000まで延長すると旧方式648000回、新方式16000回。これはforward反復数の比で、
   training全体の40.5倍高速化を意味しません。
2. cross OTの128×32768行列はfloat64で32 MiB。各反復で複数回この行列を読み書きし、
   logsumexpを行います。target自己OTの32768²行列は作っていません。
3. 全反復を微分し、10反復ごとにactivation checkpointingしています。
   backward時にも再計算が入るため、forward反復の削減だけでは総時間は同じ比率で減りません。
   PyTorch公式もcheckpointingをmemoryとcomputeの交換として説明しています。
   [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
4. PCAは一度fitしreal側もcache済み。predictionは128×50へ射影した後にOTへ渡します。
   targetのCPU→GPU転送は32768×50×4 bytes=6.25 MiB/step。
   source/target除外、numpy sampling、hashingもありますが、GPU実測なしでは寄与を断定できません。
5. CellUNetは固定で、通常のsource inputも勾配を持たないため、そのforwardに
   学習用のCellUNet backwardは発生しません。追加のfreezeで得られる改善はありません。
6. centered/shifted Hill自体もcell×gene×geneの中間計算を持ちます。
   OTだけが全所要時間を占めるとは断定できません。sampling・UMAPも各条件の後に直列実行します。

## CPUで比較した結果

環境: arm64 CPU、PyTorch 2.5.1、Python 3.9.23、1 thread、float64。
各方式2試行。下表は初回のライブラリ初期化を避けて2回目を記載しています。
試行数が少なく、速度は参考値です。GPU128×32768の性能推定には使いません。
一つのrectangular OTのforward+backwardで、ODE・自己OT・samplingは含みません。

| shape (source,target,dim) | 旧再試行 | 継続方式 | epsilon-scaling候補 |
|---|---:|---:|---:|
| 4,7,3 | 0.1255秒 | 0.0724秒 | 0.0354秒 |
| 8,512,50 | 0.1471秒 | 0.1058秒 | 0.1211秒 |

旧再試行と継続方式は、両方のfixtureでloss差0・勾配差0。
epsilon-scalingは同じ最終epsilon=0.1、marginal tolerance=1e-5でも、継続方式に対し
勾配の相対L2差がそれぞれ0.00313、0.00152でした。最終目的関数は同じでも、有限精度・
有限反復での勾配経路が異なります。全条件での速度向上と勾配一致が確認できないため、
既定の学習には採用していません。

生データ: [small](benchmark_cpu_small.json)、[rectangular](benchmark_cpu_rectangular.json)。

実装検証: unittest 20件が10.345秒で成功。全4ODEへのgradient、32768 target、
既知source変更からの再開、途中EMAへの上限短縮、元config/checkpointの不変、
短縮前後のevaluation marker分離を含みます。実データ/GPUでの本学習は未実行です。

## 採用した変更

- 途中状態を保持して200反復ずつ延長する既存修正を継続使用。
- random演算を含まないSinkhorn blockではcheckpointのRNG保存を省略。
  全反復の微分・float64・epsilon・許容誤差は保持し、従来solverとの厳密な勾配一致をテスト。
- 毎200反復の大量stdout出力を既定で止め、50stepごとの学習進捗に集約。
  cross/selfそれぞれの反復数・残差・延長履歴は全stepのCSVに保存。
- UTC timestamp・秒/step・残り時間の表示。各起動の最初の3更新はCUDAを同期し、
  data / forward / backward / updateを`timing.jsonl`に保存。
- Stage2上限を10000へ短縮。LR annealingは元の30000step scheduleを維持。
  保存済み設定・30k結果を上書きせず、reconstructionは10k時点のEMAを再利用して再評価。
  OTは10k以下の最新完全bundleから再開し10kで停止。新しい比較表も10kの解析markerを参照。

## 次に検討できる候補と条件

| 候補 | 狙い | 現時点での判断 |
|---|---|---|
| epsilon-scaling / dual warm start | 収束までの反復を減らす | 小規模比較を実施。実データでの収束・勾配の確認が必要 |
| envelope theorem / implicit differentiation | 何千反復ものbackwardを省く | 有力。ただし有限反復をそのまま微分する現方式と差が出得るため未採用 |
| KeOps / GeomLoss | GPUでのpairwise計算・memory改善 | 新依存と数値定義の対応付けが必要。128×32768ではmatrix不要化だけで必ず速くなるとは限らない |
| torch.compile / fused logsumexp | kernel起動や中間行列の読み書きを削減 | GPU型番・PyTorch版を確認してbenchmarkが必要 |
| PCA targetのGPU cache | CPU gather/転送の削減 | 追加GPUメモリが必要。data時間が支配的なら検討 |
| target数・PCA次元・epsilon・toleranceの変更 | 問題を軽くする | 実験条件が変わるため、今回の高速化では変更しない |

GeomLossはepsilon-scalingやKeOps backend、収束点での明示勾配を利用します。
同じ名前のSinkhornでもdefault cost normalization、blurとepsilon、debias、収束条件を
対応させる必要があります。PCA50Dに低次元向けmultiscaleの速度をそのまま期待はできません。
参照: [GeomLoss](https://www.kernel-operations.io/geomloss/)、
[API](https://www.kernel-operations.io/geomloss/api/pytorch-api.html)、
[gradient / truncation explanation](https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_kernel_truncation.html)、
[implicit Sinkhorn differentiation paper](https://arxiv.org/abs/2205.06688)。

## 時間の見積もり

旧速度20.76秒/stepが変わらない仮定では、2536→10000は約43.0時間。
実際の再開は最新完全checkpointからなので、2000からなら約46.1時間です。
継続方式の速度はリモート実測後に計算し直す必要があります。
他モデル3つの数十stepの失敗runの速度は、長時間学習の予測に使いません。

## GPUでの追加測定（任意、学習を更新しない）

まず通常再開の`[timing]`でforward/backward比を確認します。
GPUでsolver単体を測る場合は競合する学習を止めてから、以下を実行できます。
1回の128×32768 OTだけでも重くなり得るため、既定の小規模CPU試験とは区別します。

```bash
python -m work.20260916_x0predict_hybrid_additive.scripts.benchmark_ot \
  --device cuda:0 --sources 128 --targets 32768 --dimension 50 \
  --scale 5 --repeats 2 --max-iterations 16000 \
  --methods continuation epsilon_scaling_candidate
```

乱数fixtureであり実データそのもののbenchmarkではありません。
