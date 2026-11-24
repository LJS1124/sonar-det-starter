"""数据统计（第 1–2 天）：类别分布、框大小、目标在距离方向上的位置、图像亮度。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import range_fraction
from .io import read_gray
from .uatd import Annotation

# 按 COCO 的习惯用「面积开方」分大中小，但声呐图通常比 COCO 图小，所以阈值按像素自己设
SIZE_BINS = [0, 16, 32, 64, 1e9]
SIZE_LABELS = ["<16px", "16-32px", "32-64px", ">=64px"]
RANGE_BINS = [0, 1 / 3, 2 / 3, 1.0001]
RANGE_LABELS = ["近", "中", "远"]


def boxes_table(anns: list[Annotation], near_at_top: bool = True) -> pd.DataFrame:
    rows = []
    for a in anns:
        h = a.height or read_gray(a.image_path).shape[0]
        for b in a.boxes:
            rows.append({
                "image": a.stem, "cls": b.cls, "w": b.w, "h": b.h,
                "sqrt_area": float(np.sqrt(max(b.w * b.h, 0))),
                "aspect": b.w / max(b.h, 1e-6),
                "range_frac": range_fraction(b.cy, h, near_at_top),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["size_bin"] = pd.cut(df["sqrt_area"], SIZE_BINS, labels=SIZE_LABELS, right=False)
        df["range_bin"] = pd.cut(df["range_frac"], RANGE_BINS, labels=RANGE_LABELS, right=False)
    return df


def image_table(anns: list[Annotation], sample: int | None = 300, seed: int = 0) -> pd.DataFrame:
    """每张图的亮度统计。dark_ratio 是「暗到几乎看不见」的像素比例。"""
    rng = np.random.default_rng(seed)
    pick = anns if sample is None or len(anns) <= sample else [anns[i] for i in rng.choice(len(anns), sample, replace=False)]
    rows = []
    for a in pick:
        img = read_gray(a.image_path)
        x = img.astype(np.float32)
        full = 65535.0 if img.dtype == np.uint16 else 255.0
        rows.append({
            "image": a.stem, "dtype": str(img.dtype), "height": img.shape[0], "width": img.shape[1],
            "mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()),
            "dark_ratio": float((x < 0.05 * full).mean()),
            "n_boxes": len(a.boxes),
        })
    return pd.DataFrame(rows)
