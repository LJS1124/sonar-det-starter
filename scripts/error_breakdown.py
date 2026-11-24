"""第 9–10 天：按目标大小、距离、类别拆解召回率，并统计每张图的误报数。

先用模型对测试集出预测（保存 txt 和置信度）：
    yolo predict model=runs/<name>/weights/best.pt source=data/yolo_xxx/images/test \\
        save_txt=True save_conf=True conf=0.001 project=runs name=<name>_pred exist_ok=True
再运行：
    python scripts/error_breakdown.py --yolo-dir data/yolo_xxx --split test \\
        --pred-dir runs/<name>_pred/labels --conf 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonar.breakdown import false_positives_per_image, match_dataset, recall_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="reports/breakdown")
    args = ap.parse_args()

    img_dir = Path(args.yolo_dir) / "images" / args.split
    sizes = {}
    for p in sorted(img_dir.glob("*.png")):
        h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape[:2]
        sizes[p.stem] = (w, h)
    gt_df, fp_df = match_dataset(Path(args.yolo_dir) / "labels" / args.split, Path(args.pred_dir),
                                 sizes, conf=args.conf, iou_thr=args.iou)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(out / "gt_matches.csv", index=False)
    fp_df.to_csv(out / "false_positives.csv", index=False)
    print(f"总召回率：{gt_df['detected'].mean():.3f}（{len(gt_df)} 个真值框）")
    print(f"每张图平均误报：{false_positives_per_image(fp_df, len(sizes)):.2f}")
    for by in ("size_bin", "range_bin", "cls"):
        print(f"\n== 按 {by} ==\n{recall_table(gt_df, by).to_string()}")


if __name__ == "__main__":
    main()
