"""极坐标（距离 × 方位）和扇形显示之间的转换。

前视声呐的原始数据是一个矩形矩阵：一个轴是距离单元（range bin），另一个轴是波束（方位角）。
人眼习惯看的「扇形图」是把这个矩阵按真实几何展开的结果。

约定（第 1 天要用真实数据核对）：
- 行（row）= 距离：row 0 是离声呐最近的地方，row H-1 是最大量程；
- 列（col）= 方位：col 0 是最左边的波束，col W-1 是最右边的波束。
如果你的数据方向相反，用 np.flipud / np.fliplr 先翻过来。
"""
from __future__ import annotations

import cv2
import numpy as np


def polar_to_fan(
    polar: np.ndarray,
    fov_deg: float = 120.0,
    out_height: int | None = None,
    near_at_top: bool = True,
) -> np.ndarray:
    """把极坐标矩阵展开成扇形图，声呐位于图像底部中央。

    fov_deg：水平开角。Gemini 1200ik 的开角请以 XML 里的方位参数或设备手册为准。
    """
    h, w = polar.shape[:2]
    if not near_at_top:
        polar = np.flipud(polar)
    out_h = out_height or h
    half = np.deg2rad(fov_deg / 2)
    out_w = int(np.ceil(2 * out_h * np.sin(half))) + 1

    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    dx = xs - out_w / 2
    dy = out_h - ys                      # 声呐在底部中央，向上为正前方
    r = np.sqrt(dx**2 + dy**2) / out_h   # 归一化距离 0–1
    theta = np.arctan2(dx, dy)           # 正前方为 0，向右为正

    map_y = (r * (h - 1)).astype(np.float32)
    map_x = ((theta + half) / (2 * half) * (w - 1)).astype(np.float32)
    valid = (r <= 1.0) & (np.abs(theta) <= half)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return cv2.remap(polar, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def range_fraction(cy: float, height: int, near_at_top: bool = True) -> float:
    """框中心在距离方向上的位置，0 = 最近，1 = 最大量程。乘以量程就是距离（米）。"""
    f = cy / max(height - 1, 1)
    return f if near_at_top else 1 - f
