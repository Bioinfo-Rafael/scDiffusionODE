# ファイル整理用
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
import re
from turtle import color
from typing import Dict, Tuple, List


# -----------------------------
# ここをあなたの対応表に合わせる
# -----------------------------
# key: フォルダ名（ファイル実行条件）
# val: (学習前後, 配色)
CONDITION_MAP: Dict[str, Tuple[str, str]] = {
    "20260305_044239_velocity_jupyter": ("学習後", "ランダム配色"),
    "20260305_053513_velocity_jupyter": ("学習後", "グラデーション配色"),
    "20260305_112258_velocity_jupyter": ("学習前", "ランダム配色"),
    "20260305_113413_velocity_jupyter": ("学習前", "グラデーション配色"),
}

# 実行条件フォルダ名からID部分を抜き出す（例: 20260305_044239）
RUN_ID_RE = re.compile(r"^(?P<runid>\d{8}_\d{6})_velocity_jupyter$")


@dataclass
class PlanItem:
    src: Path
    dst: Path
    condition_dir: str
    run_id: str
    training: str
    color: str
    cell: str
    image_type: str
    action: str  # "copy" or "move"


def build_plan(src_root: Path, dst_root: Path, move: bool) -> List[PlanItem]:
    plan: List[PlanItem] = []
    action = "move" if move else "copy"

    for cond_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        cond_name = cond_dir.name
        if cond_name not in CONDITION_MAP:
            print(f"[WARN] CONDITION_MAPに無いのでスキップ: {cond_name}")
            continue

        m = RUN_ID_RE.match(cond_name)
        if not m:
            print(f"[WARN] フォルダ名が想定形式でないのでスキップ: {cond_name}")
            continue

        run_id = m.group("runid")
        training, color = CONDITION_MAP[cond_name]

        # 細胞集団フォルダ
        for cell_dir in sorted([p for p in cond_dir.iterdir() if p.is_dir()]):
            cell = cell_dir.name

            # 画像ファイル（png/jpg/jpeg/tif/tiff 等を対象）
            for img in sorted([p for p in cell_dir.iterdir() if p.is_file()]):
                if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    continue

                image_type = img.stem  # 例: "1_velocity_stream"
                # 出力先: 細胞集団/配色/画像タイプ/学習前・後/
                # out_dir = dst_root / cell / color / image_type / training
                out_dir = dst_root / cell / image_type / training / color
                # リネーム: 元stem + _run_id_学習前(後) + 拡張子
                new_name = f"{img.stem}_{run_id}_{training}{img.suffix}"
                dst = out_dir / new_name

                plan.append(
                    PlanItem(
                        src=img,
                        dst=dst,
                        condition_dir=cond_name,
                        run_id=run_id,
                        training=training,
                        color=color,
                        cell=cell,
                        image_type=image_type,
                        action=action,
                    )
                )

    return plan


def ensure_parent(p: Path, dry_run: bool) -> None:
    if dry_run:
        return
    p.parent.mkdir(parents=True, exist_ok=True)


def execute_plan(plan: List[PlanItem], dry_run: bool) -> None:
    # 同名衝突を回避（まれに同じdstが出た場合）
    used = set()

    for item in plan:
        dst = item.dst
        if dst in used or dst.exists():
            # 末尾に連番を付ける
            base = dst.with_suffix("")
            suf = dst.suffix
            i = 2
            while True:
                cand = Path(str(base) + f"__{i}").with_suffix(suf)
                if (cand not in used) and (not cand.exists()):
                    dst = cand
                    break
                i += 1

        used.add(dst)

        if dry_run:
            print(f"[DRY] {item.action.upper()}  {item.src} -> {dst}")
            continue

        ensure_parent(dst, dry_run=False)
        if item.action == "move":
            shutil.move(str(item.src), str(dst))
        else:
            shutil.copy2(str(item.src), str(dst))


def write_log(plan: List[PlanItem], log_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] ログ出力予定: {log_path}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "action", "src", "dst",
            "condition_dir", "run_id", "training", "color", "cell", "image_type"
        ])
        for it in plan:
            w.writerow([
                it.action, str(it.src), str(it.dst),
                it.condition_dir, it.run_id, it.training, it.color, it.cell, it.image_type
            ])


def main():
    ap = argparse.ArgumentParser(description="velocity_jupyter画像を並べ替えてリネームする")
    ap.add_argument("--src", required=True, help="入力ルート (例: C:\\Users\\rafae\\Downloads\\velocity_jupyter_all)")
    ap.add_argument("--dst", default="", help="出力ルート (省略時: <src>_organized)")
    ap.add_argument("--move", action="store_true", help="コピーではなく移動にする（元を消したい場合）")
    ap.add_argument("--dry-run", action="store_true", help="実際には作業せず、予定だけ表示")
    args = ap.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"srcが存在しません: {src_root}")

    dst_root = Path(args.dst).expanduser().resolve() if args.dst else Path(str(src_root) + "_organized").resolve()

    plan = build_plan(src_root, dst_root, move=args.move)

    print("---- PLAN SUMMARY ----")
    print(f"src : {src_root}")
    print(f"dst : {dst_root}")
    print(f"mode: {'MOVE' if args.move else 'COPY'}")
    print(f"items: {len(plan)}")
    print("----------------------")

    if len(plan) == 0:
        print("[INFO] 対象ファイルが見つかりませんでした。CONDITION_MAPや拡張子を確認してください。")
        return

    # 実行（dry-runなら表示だけ）
    execute_plan(plan, dry_run=args.dry_run)

    # ログ
    log_path = dst_root / "_reorg_log.csv"
    write_log(plan, log_path, dry_run=args.dry_run)

    print("DONE.")
    if args.dry_run:
        print("※ dry-run なのでファイル操作はしていません。")
    else:
        print(f"ログ: {log_path}")


if __name__ == "__main__":
    main()