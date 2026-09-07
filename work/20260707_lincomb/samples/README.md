# 2026-07-07 sampling出力

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連するsampling出力の保存先です。training runとは分離して管理し、手動編集しません。

```text
samples/
  <experiment>/
    <run_id>/
      samples.npz
      sample_manifest.json
```

downstream visualizationはmanifestに記録されたsample pathを参照します。`hybrid_ts_soft_lincomb` のsampling時は、trainingで解決済みの `T_s` とgate設定をconfigから復元します。

単独実行例:

```bash
python work/20260707_lincomb/scripts/sample_0707.py \
  --run-dir work/20260707_lincomb/runs/<experiment>/<run_id>
```

