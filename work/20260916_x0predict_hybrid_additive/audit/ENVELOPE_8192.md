# 2026-09-17: Envelope勾配 + 8192 targetへの継続

ユーザーの明示指示により、既存checkpointからの途中切替を実装しました。
Stage2は10kを維持し、旧32768/full-autograd履歴の後から8192/envelopeへ移行します。
これは最初から新方式で学習した実験ではありません。

## 数値定義

固定PCA50、mean-squared cost、epsilon 0.1、uniform marginals、float64、
absolute marginal tolerance 1e-5を維持します。初回200反復、以後200ずつ延長し
最大16000。10反復ごとに収束を検査し、途中状態を保持します。
Envelopeでもforwardの収束計算は必要です。

cost Cはpredictionに対して微分可能なまま作成します。dual解u,vとplanを
`torch.no_grad()`で求め、`π=exp(-C/epsilon+u+v)`を固定したうえで
`sum(π*dC)`をprediction→PCA→ODEへ流します。loss表示にはKL項も含めます。
prediction自己OTでは両方の引数への勾配を含めます。target-only自己項は従来どおり
モデルに依存しない定数として省略し、8192²の自己行列は作りません。

収束解でのenvelope勾配と、有限反復を自動微分する旧方式の勾配は
有限の停止精度では差が出得ます。forward値と停止反復数は同じ入力で一致します。
参考: [GeomLossの収束点における明示勾配](https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_kernel_truncation.html)。

## 再開と処理順

1. 共通CellUNet baselineの完了結果を再利用。
2. centered_hill_reconst → shifted_hill_reconst → hill_after_linear_reconst → softplus_reconst。
3. centered_hill_ot → shifted_hill_ot → hill_after_linear_ot → softplus_ot。

各条件ごとにtrain→sample→数値analysis/plot→UMAP/plotを完了して次へ進みます。
数値解析失敗時もUMAPを試し、条件失敗時には独立した次条件へ進む既存方針を維持します。
reconstructionは完了した10k結果を再利用します。

旧configは書き換えません。OTの新しいmarkerには`n8192_envelope`を付け、旧結果を
新方式の完了結果として扱いません。raw/EMA/optimizerの最新完全bundleから再開し、
AdamWのmoment・update count・EMA・learning-rate scheduleを引き継ぎます。
`ot_transitions`には変更前後の設定、source checkpoint/hash、最初の新方式updateを保存します。
sampling metadataと比較表にも履歴を引き継ぎます。

同じ8192/envelopeのcheckpointがすでにあればその枝の続きを優先します。
旧方式が10kまで完了していた場合は10k未満の最後のbundleから再開します。
例えば9000が最後なら、9001〜10000だけ新方式で学習します。
新方式の更新が一度もない旧10k checkpointを、新方式で学習したものとして付け替えません。

## CPUベンチマーク

CPU arm64 / PyTorch 2.5.1 / float64 / 1 thread。
source 8・PCA50の乱数fixture（scale 2）による、一つのcross OT forward+backwardです。
各方式2試行中、初回のライブラリ初期化を避けた2回目を示します。

| target | 全反復autograd | Envelope | 同じtargetでの比 |
|---:|---:|---:|---:|
| 32768 | 0.5556秒 | 0.2094秒 | 約2.65倍 |
| 8192 | 0.1080秒 | 0.0380秒 | 約2.84倍 |

同一target内ではloss差0、停止反復数も同じです。
Envelopeの勾配相対L2差は32768で約6.89e-5、8192で約1.58e-4でした。
raw結果: [32768](benchmark_envelope_32768.json)、[8192](benchmark_envelope_8192.json)。

target削減と方式変更を合わせたこのfixtureのcross OT時間差は約14.6倍ですが、
実データ・source128・GPUでのtraining全体が14.6倍速くなるという意味ではありません。
CellUNet/ODE forward、自己OT、optimizer、sampling/UMAPの時間は含んでいません。
リモート実行時の`[timing]`・`seconds_per_step`・`eta_hours`で実速度を確認します。

## 検証範囲

2026-09-17、変更コミットの独立したclean checkoutで24テストすべて成功
（Python 3.9.23 / PyTorch 2.5.1 CPU、10.833秒）。作業ツリーにある対象外の
未コミット変更は検証用checkoutにもコミットにも含めていません。

- Envelopeで反復checkpoint graphが作られないこと。
- 同一入力でのfull-autogradとのforward値・反復数の一致。
- 厳しい収束許容誤差での有限差分と勾配の一致。prediction自己項の両引数も対象。
- 4種類すべてのODEへPCA/OT data-gradientが流れ、CellUNetが固定されること。
- 8192件・32768件のtarget除外とrectangular OT、refresh/衝突補充。
- 旧方式から新方式への切替、AdamW update count継続、source checkpointの不変。
- reconstruction優先かつ条件ごとのtrain/sample/analysis/UMAP順。
- 停止helperのcampaign選別、子孫処理、PID再利用への対処（signalはmockで検証）。

実GPUでの速度・メモリ、および生成品質はローカルでは未検証です。
