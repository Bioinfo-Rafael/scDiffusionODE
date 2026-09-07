# 2026-07-07 実行log

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連するstdout/stderrとstage logの保存先です。生成物なので手動編集しません。

```text
logs/<experiment>/<run_id>/
  train/
  sample/
  viz/
```

長時間学習の進捗、loss、checkpoint保存状況を確認するときに使います。failed stageの原因は、対応するlogと `runs/<experiment>/<run_id>/manifest.json` のstatus・return codeを合わせて確認してください。

`hybrid_ts_soft_lincomb` では、training logにAnnData読込、`T_s` 解決、model buildの流れが記録されます。

