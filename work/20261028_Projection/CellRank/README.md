# CellRank entry point

`projector.py:CellRankProjector.fit_scalar()` に `(x,v)→P→terminal→T→τ→連続近似` の順序を明記する。`cellrank_core.py` は公式APIを呼ぶwrapperと結果検査だけ。

1. `scanpy.pp.neighbors(use_rep='X')`
2. `VelocityKernel(xkey='projection_expression', vkey='projection_velocity', gene_subset=ordered_IDs)`
3. `.compute_transition_matrix(model='deterministic', similarity='cosine')`
4. `GPCCA(...).set_terminal_states({name: cell_IDs})`
5. `.compute_fate_probabilities(keys=None)`
6. `.compute_absorption_times(keys=None)`

採用: **CellRank 2.0.7**。全競合terminalを保ったまま吸収計算し、DataFrameの `<lineage>_mean` を選ぶ。複数terminalのTは**選択lineageへ吸収される条件での平均ステップ数**。対象以外のterminalのT=infは公式の結果で、勝手に有限値へ置換しない。finite Tかつ選択lineageの確率>=1e-6でfitする。メイン連続列は全細胞で評価するため、eligible=Falseでの値は外挿であり解釈対象から区別する。

定数T、対象欠如、terminalへの到達不能、公式API/solver失敗、補間誤差超過は停止する。ConnectivityKernelとの暗黙混合・独自MFPT solver・自動terminal推定はない。`solver='direct'` は2,000細胞以下だけ。

原論文: https://www.nature.com/articles/s41592-024-02303-9

公式: https://github.com/scverse/cellrank/tree/v2.0.7

`τ=1−(T−T_min)/(T_max−T_min)`、連続拡張、等値面射影は今回追加した処理。名称はCellRank absorption-time-derived pseudotimeであり、PseudotimeKernelや実時間ではない。式・検証・コマンド・保存形式は [共通README](../README.md) を参照する。
