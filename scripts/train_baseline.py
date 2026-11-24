"""第 3–4 天（及之后每个实验）：用 Ultralytics YOLO 训练，并在验证集和测试集上评估。

光学默认设置（对照组）：
    python scripts/train_baseline.py --data data/yolo_raw/data.yaml --name baseline_optical
声呐设置（关掉颜色增强）：
    python scripts/train_baseline.py --data data/yolo_log_clahe/data.yaml --name log_clahe --sonar-aug

所有实验的结果追加写入 reports/experiments.csv，方便最后汇总成对比表。
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt", help="预训练权重，如 yolo11n.pt / yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sonar-aug", action="store_true",
                    help="关掉 HSV 颜色增强；声呐是单通道强度，颜色增强没有物理意义")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO  # 延迟导入，方便没装 ultralytics 时也能看 --help

    model = YOLO(args.model)
    train_kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        project="runs", name=args.name, seed=args.seed, deterministic=True,
                        patience=30, exist_ok=True)
    if args.sonar_aug:
        train_kwargs.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    model.train(**train_kwargs)

    best = Path("runs") / args.name / "weights" / "best.pt"
    model = YOLO(str(best))
    row = {"time": datetime.now().isoformat(timespec="seconds"), "name": args.name,
           "data": args.data, "model": args.model, "sonar_aug": args.sonar_aug}
    import yaml
    has_test = "test" in yaml.safe_load(Path(args.data).read_text())
    for split in ["val"] + (["test"] if has_test else []):
        m = model.val(data=args.data, split=split, imgsz=args.imgsz, project="runs",
                      name=f"{args.name}_{split}", exist_ok=True)
        row[f"{split}_mAP50"] = round(float(m.box.map50), 4)
        row[f"{split}_mAP50_95"] = round(float(m.box.map), 4)
        print(f"[{split}] mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    csv_path = out / "experiments.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
