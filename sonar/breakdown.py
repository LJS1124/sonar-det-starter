"""误差拆解（第 9–10 天）：总 mAP 之外，按目标大小和距离看召回率，按图看误报数。

输入是 YOLO 格式的预测文件（每行：class cx cy w h conf，归一化坐标），
用 Ultralytics 预测时加 save_txt=True, save_conf=True 就能得到。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import RANGE_BINS, RANGE_LABELS, SIZE_BINS, SIZE_LABELS


def _read_yolo(path: Path, with_conf: bool) -> np.ndarray:
    cols = 6 if with_conf else 5
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, cols), dtype=np.float32)
    arr = np.loadtxt(path, ndmin=2, dtype=np.float32)
    return arr[:, :cols]


def _to_xyxy(a: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_dataset(gt_dir: Path, pred_dir: Path, sizes: dict[str, tuple[int, int]],
                  conf: float = 0.25, iou_thr: float = 0.5, near_at_top: bool = True):
    """逐图贪心匹配（同类别、IoU ≥ iou_thr，按置信度从高到低）。

    sizes：{图像名: (宽, 高)}，用来把归一化坐标换回像素。
    返回 (gt_df, fp_df)：gt_df 每行一个真值框及是否被检出；fp_df 每行一个误报。
    """
    gt_rows, fp_rows = [], []
    for stem, (w, h) in sizes.items():
        gt = _read_yolo(Path(gt_dir) / f"{stem}.txt", with_conf=False)
        pr = _read_yolo(Path(pred_dir) / f"{stem}.txt", with_conf=True)
        pr = pr[pr[:, 5] >= conf] if len(pr) else pr
        pr = pr[np.argsort(-pr[:, 5])] if len(pr) else pr
        g_xy, p_xy = _to_xyxy(gt, w, h), _to_xyxy(pr, w, h)
        ious = _iou(p_xy, g_xy)
        matched = np.zeros(len(gt), dtype=bool)
        for i in range(len(pr)):
            cand = np.where((gt[:, 0] == pr[i, 0]) & ~matched)[0] if len(gt) else np.array([], int)
            j = cand[np.argmax(ious[i, cand])] if len(cand) else -1
            if j >= 0 and ious[i, j] >= iou_thr:
                matched[j] = True
            else:
                fp_rows.append({"image": stem, "cls": int(pr[i, 0]), "conf": float(pr[i, 5]),
                                "range_frac": float(pr[i, 2] if near_at_top else 1 - pr[i, 2])})
        for k in range(len(gt)):
            bw, bh = gt[k, 3] * w, gt[k, 4] * h
            gt_rows.append({"image": stem, "cls": int(gt[k, 0]), "detected": bool(matched[k]),
                            "sqrt_area": float(np.sqrt(bw * bh)),
                            "range_frac": float(gt[k, 2] if near_at_top else 1 - gt[k, 2])})
    gt_df, fp_df = pd.DataFrame(gt_rows), pd.DataFrame(fp_rows)
    if len(gt_df):
        gt_df["size_bin"] = pd.cut(gt_df["sqrt_area"], SIZE_BINS, labels=SIZE_LABELS, right=False)
        gt_df["range_bin"] = pd.cut(gt_df["range_frac"], RANGE_BINS, labels=RANGE_LABELS, right=False)
    return gt_df, fp_df


def recall_table(gt_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """按 by（size_bin / range_bin / cls）分组的召回率。"""
    g = gt_df.groupby(by, observed=False)["detected"]
    return pd.DataFrame({"n_gt": g.size(), "recall": g.mean().round(3)})


def false_positives_per_image(fp_df: pd.DataFrame, n_images: int) -> float:
    return float(len(fp_df) / max(n_images, 1))
