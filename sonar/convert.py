"""转换成 Ultralytics YOLO 格式（第 3 天），也用于生成预处理 / 增强后的数据变体（第 5–8 天）。

输出结构：
    out_dir/
      images/{train,val,test}/xxx.png
      labels/{train,val,test}/xxx.txt     每行：class cx cy w h（归一化到 0–1）
      data.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .io import read_gray, to_uint8, write_image
from .uatd import Annotation


def yolo_lines(ann: Annotation, classes: list[str], width: int, height: int) -> list[str]:
    lines = []
    for b in ann.boxes:
        if b.cls not in classes:
            continue
        x1, y1 = max(0.0, b.x1), max(0.0, b.y1)
        x2, y2 = min(float(width), b.x2), min(float(height), b.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{classes.index(b.cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def write_split(
    anns: list[Annotation],
    split: str,
    out_dir: Path,
    classes: list[str],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    augment: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    aug_copies: int = 0,
    seed: int = 0,
) -> int:
    """写出一个划分。transform 是预处理（所有图都做）；augment 只用于额外生成的副本。

    增强副本只应该用在训练集上，验证集和测试集保持原样。
    """
    rng = np.random.default_rng(seed)
    n = 0
    for ann in anns:
        img = read_gray(ann.image_path)
        h, w = img.shape[:2]
        base = transform(img) if transform else to_uint8(img)
        lines = yolo_lines(ann, classes, w, h)
        variants = [("", base)]
        if augment is not None:
            variants += [(f"_aug{k}", augment(base, rng)) for k in range(aug_copies)]
        for suffix, im in variants:
            name = f"{ann.stem}{suffix}"
            write_image(out_dir / "images" / split / f"{name}.png", im)
            lbl = out_dir / "labels" / split / f"{name}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    return n


def write_data_yaml(out_dir: Path, classes: list[str], has_test: bool) -> Path:
    cfg = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)},
    }
    if has_test:
        cfg["test"] = "images/test"
    p = Path(out_dir) / "data.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return p
