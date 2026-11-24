"""图像读写。声呐图可能是 8 位，也可能是 16 位，一律按原始位深读进来。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    """读成单通道，保留原始位深（uint8 或 uint16）。

    有的声呐图存成了 3 通道但三个通道相同，这里取第一个通道。
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    """线性缩放到 0–255，仅用于显示或保存，不做任何增强。"""
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"写入失败：{path}")
