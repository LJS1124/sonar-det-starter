"""声呐预处理（第 5–6 天）。

为什么需要：原始回波强度的动态范围很大，少数强回波（目标正面、旁瓣亮线）很亮，
大部分区域接近全黑。直接线性缩放到 8 位，弱目标会被压成一片黑。

每个函数输入任意位深的单通道图，输出 uint8，方便直接喂给检测器。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from .io import to_uint8


def log_compress(img: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """对数压缩：把乘性的强度差异变成加性的，弱回波被拉起来，强回波被压下去。"""
    x = np.log(img.astype(np.float32) + eps)
    return to_uint8(x)


def percentile_stretch(img: np.ndarray, lo: float = 1.0, hi: float = 99.8) -> np.ndarray:
    """按分位数截断再拉伸。比 min-max 稳定，不会被个别极亮像素带偏。"""
    x = img.astype(np.float32)
    a, b = np.percentile(x, [lo, hi])
    if b <= a:
        return to_uint8(x)
    return (np.clip((x - a) / (b - a), 0, 1) * 255).astype(np.uint8)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """局部直方图均衡。注意：它也会放大斑点噪声，所以通常放在去斑之后。"""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return op.apply(to_uint8(img))


def lee_filter(img: np.ndarray, size: int = 5, looks: float = 1.0) -> np.ndarray:
    """Lee 滤波：经典的乘性斑点噪声抑制方法（来自 SAR 图像处理，声呐同样适用）。

    思路：在每个窗口里比较局部方差和「纯噪声应有的方差」。
    方差接近噪声水平 → 认为是平坦区域，用均值平滑；方差明显更大 → 认为是边缘或目标，保留原值。
    looks：等效视数，越大表示噪声越弱。
    """
    x = img.astype(np.float32)
    mean = uniform_filter(x, size)
    mean_sq = uniform_filter(x * x, size)
    var = np.maximum(mean_sq - mean * mean, 0)
    noise_cv2 = 1.0 / looks                     # 斑点噪声的变异系数平方
    signal_var = np.maximum(var - noise_cv2 * mean * mean, 0)
    k = signal_var / (var + 1e-6)
    out = mean + k * (x - mean)
    return to_uint8(out) if img.dtype != np.uint8 else np.clip(out, 0, 255).astype(np.uint8)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(to_uint8(img), ksize)


PIPELINES = {
    # 对照组：只做线性缩放，相当于「当成光学图」
    "raw": lambda x: to_uint8(x),
    "stretch": lambda x: percentile_stretch(x),
    "log": lambda x: log_compress(x),
    "log_clahe": lambda x: clahe(log_compress(x)),
    "lee_log_clahe": lambda x: clahe(log_compress(lee_filter(x))),
    "median_log_clahe": lambda x: clahe(log_compress(median_denoise(x))),
}


def sonar_preprocess(img: np.ndarray, mode: str = "log_clahe") -> np.ndarray:
    if mode not in PIPELINES:
        raise KeyError(f"未知预处理 {mode!r}，可选：{list(PIPELINES)}")
    return PIPELINES[mode](img)
