"""可视化：画标注框、拼图。"""
from __future__ import annotations

import cv2
import numpy as np

from .io import read_gray, to_uint8
from .uatd import Annotation

_PALETTE = [(255, 99, 71), (60, 179, 113), (65, 105, 225), (238, 130, 238), (255, 215, 0),
            (0, 206, 209), (255, 140, 0), (154, 205, 50), (199, 21, 133), (176, 196, 222)]


def draw_boxes(ann: Annotation, img: np.ndarray | None = None, classes: list[str] | None = None,
               thickness: int = 1) -> np.ndarray:
    """返回一张 RGB 图。img 不传则读原图并线性缩放到 8 位（不做增强，看到的就是「原样」）。"""
    if img is None:
        img = to_uint8(read_gray(ann.image_path))
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    for b in ann.boxes:
        k = classes.index(b.cls) if classes and b.cls in classes else sum(map(ord, b.cls))
        color = _PALETTE[k % len(_PALETTE)]
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(rgb, p1, p2, color, thickness)
        cv2.putText(rgb, b.cls, (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return rgb


def montage(images: list[np.ndarray], cols: int = 4, cell: tuple[int, int] = (256, 256)) -> np.ndarray:
    """把若干张图缩放到相同大小后拼成网格。"""
    cw, ch = cell
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i, im in enumerate(images):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = im
    return canvas
