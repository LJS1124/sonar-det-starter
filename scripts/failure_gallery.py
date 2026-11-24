"""第 9–10 天：把漏检和高置信度误报裁出来拼成图集，方便逐个标注失败原因。

先运行 error_breakdown.py 生成 reports/breakdown/*.csv，再运行：
    python scripts/failure_gallery.py --yolo-dir data/yolo_xxx --split test \\
        --pred-dir runs/<name>_pred/labels --breakdown reports/breakdown --n 12
输出 failure_missed.png、failure_false_pos.png，以及一个待填写原因的 failure_cases.csv。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonar.viz import montage  # noqa: E402


def _crop(img: np.ndarray, cx: float, cy: float, bw: float, bh: float, color, pad: float = 2.0) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    s = max(bw, bh) * pad + 24
    cx1, cy1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    cx2, cy2 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    return rgb[cy1:cy2, cx1:cx2]


def _find_label(lbl_path: Path, cls: int, frac_y: float, with_conf: bool):
    if not lbl_path.exists():
        return None
    rows = [list(map(float, ln.split())) for ln in lbl_path.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if int(r[0]) == cls]
    return min(rows, key=lambda r: abs(r[2] - frac_y)) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--breakdown", default="reports/breakdown")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    ydir, bdir = Path(args.yolo_dir), Path(args.breakdown)
    gt = pd.read_csv(bdir / "gt_matches.csv")
    fp = pd.read_csv(bdir / "false_positives.csv")
    missed = gt[~gt["detected"]].sample(min(args.n, int((~gt["detected"]).sum())), random_state=0)
    top_fp = fp.sort_values("conf", ascending=False).head(args.n)

    cases, tiles_m, tiles_f = [], [], []
    for kind, df, lbl_root, color, tiles in [
        ("漏检", missed, ydir / "labels" / args.split, (255, 80, 80), tiles_m),
        ("误报", top_fp, Path(args.pred_dir), (80, 200, 255), tiles_f),
    ]:
        for _, r in df.iterrows():
            img = cv2.imread(str(ydir / "images" / args.split / f"{r['image']}.png"), cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            lab = _find_label(lbl_root / f"{r['image']}.txt", int(r["cls"]), float(r["range_frac"]), kind == "误报")
            if lab is None:
                continue
            tiles.append(_crop(img, lab[1] * W, lab[2] * H, lab[3] * W, lab[4] * H, color))
            cases.append({"类型": kind, "图像": r["image"], "类别id": int(r["cls"]), "原因": ""})

    for name, tiles in [("failure_missed.png", tiles_m), ("failure_false_pos.png", tiles_f)]:
        if tiles:
            cv2.imwrite(str(bdir / name), cv2.cvtColor(montage(tiles, cols=4, cell=(160, 160)), cv2.COLOR_RGB2BGR))
    pd.DataFrame(cases).to_csv(bdir / "failure_cases.csv", index=False, encoding="utf-8-sig")
    print(f"已输出到 {bdir.resolve()}：漏检 {len(tiles_m)} 个，误报 {len(tiles_f)} 个。在 failure_cases.csv 里填写原因。")


if __name__ == "__main__":
    main()
