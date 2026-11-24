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


def fan_size(polar_shape: tuple[int, int], fov_deg: float = 120.0, out_height: int | None = None) -> tuple[int, int]:
    """polar_to_fan 输出图像的 (高, 宽)。"""
    out_h = out_height or polar_shape[0]
    return out_h, int(np.ceil(2 * out_h * np.sin(np.deg2rad(fov_deg / 2)))) + 1


def polar_points_to_fan(rows: np.ndarray, cols: np.ndarray, polar_shape: tuple[int, int],
                        fov_deg: float = 120.0, out_height: int | None = None,
                        near_at_top: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """把极坐标图上的点（行, 列）映射到扇形图上的 (x, y)，用来把标注框画到扇形图上。

    注意：极坐标里的矩形框，到了扇形图上是一个「扇环」形状，不再是矩形。
    """
    h, w = polar_shape[:2]
    out_h, out_w = fan_size(polar_shape, fov_deg, out_height)
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    r = rows / (h - 1) if near_at_top else 1 - rows / (h - 1)
    half = np.deg2rad(fov_deg / 2)
    theta = cols / (w - 1) * 2 * half - half
    x = out_w / 2 + r * out_h * np.sin(theta)
    y = out_h - r * out_h * np.cos(theta)
    return x, y


def box_to_fan_polygon(x1: float, y1: float, x2: float, y2: float, polar_shape: tuple[int, int],
                       fov_deg: float = 120.0, n: int = 20, **kw) -> np.ndarray:
    """极坐标图上的框 → 扇形图上的多边形（沿四条边采样 n 个点），形状为 (4n, 2)。"""
    t = np.linspace(0, 1, n)
    rows = np.concatenate([np.full(n, y1), y1 + (y2 - y1) * t, np.full(n, y2), y2 - (y2 - y1) * t])
    cols = np.concatenate([x1 + (x2 - x1) * t, np.full(n, x2), x2 - (x2 - x1) * t, np.full(n, x1)])
    x, y = polar_points_to_fan(rows, cols, polar_shape, fov_deg, **kw)
    return np.stack([x, y], axis=1)


def range_fraction(cy: float, height: int, near_at_top: bool = True) -> float:
    """框中心在距离方向上的位置，0 = 最近，1 = 最大量程。乘以量程就是距离（米）。"""
    f = cy / max(height - 1, 1)
    return f if near_at_top else 1 - f
