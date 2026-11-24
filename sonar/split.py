"""训练集 / 验证集划分，以及泄漏检查（第 3 天）。

声呐数据是连续采集的视频帧，相邻帧几乎一样。随机划分会让验证集里的图
在训练集里有「双胞胎」，验证分数虚高，换到新水域就掉下来。
正确做法：先按文件名自然顺序把连续帧切成块，再以块为单位划分。
"""
from __future__ import annotations

import cv2
import numpy as np

from .io import read_gray, to_uint8
from .uatd import Annotation


def random_split(anns: list[Annotation], val_ratio: float = 0.2, seed: int = 0):
    """反面教材：逐帧随机划分。只用于演示泄漏。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(anns))
    n_val = int(round(len(anns) * val_ratio))
    val = set(idx[:n_val].tolist())
    return ([a for i, a in enumerate(anns) if i not in val],
            [a for i, a in enumerate(anns) if i in val])


def block_split(anns: list[Annotation], val_ratio: float = 0.2,
                block_size: int = 50, seed: int = 0):
    """按连续块划分。anns 需要已按自然顺序排好（find_pairs 已经排好）。

    block_size 应大于「一个目标在画面里停留的帧数」。不确定的话从 50 开始，
    再用 leakage_report 检查，相似度高就调大。
    """
    blocks = [anns[i:i + block_size] for i in range(0, len(anns), block_size)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    n_val_blocks = max(1, int(round(len(blocks) * val_ratio)))
    val_ids = set(order[:n_val_blocks].tolist())
    train = [a for i, b in enumerate(blocks) if i not in val_ids for a in b]
    val = [a for i, b in enumerate(blocks) if i in val_ids for a in b]
    return train, val


def _thumb(ann: Annotation, size: int = 32) -> np.ndarray:
    img = to_uint8(read_gray(ann.image_path))
    t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def leakage_report(train: list[Annotation], val: list[Annotation],
                   threshold: float = 0.95, max_items: int = 2000, seed: int = 0) -> dict:
    """对每张验证图，找训练集里最相似的一张（缩略图的相关系数）。

    返回相似度超过 threshold 的验证图比例。这个比例越高，验证分数越不可信。
    """
    rng = np.random.default_rng(seed)
    if len(train) > max_items:
        train = [train[i] for i in rng.choice(len(train), max_items, replace=False)]
    if len(val) > max_items:
        val = [val[i] for i in rng.choice(len(val), max_items, replace=False)]
    T = np.stack([_thumb(a) for a in train])
    V = np.stack([_thumb(a) for a in val])
    best = (V @ T.T).max(axis=1)
    return {
        "n_val_checked": int(len(V)),
        "mean_best_similarity": float(best.mean()),
        "near_duplicate_ratio": float((best >= threshold).mean()),
        "threshold": threshold,
    }
