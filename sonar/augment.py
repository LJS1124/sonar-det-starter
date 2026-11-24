"""声呐数据增强（第 7–8 天）。

和光学增强的区别：
- 不做颜色、色调、饱和度增强（声呐是单通道强度）；
- 噪声用乘性的斑点噪声，不用加性高斯噪声；
- 模拟操作员调增益、调量程带来的整体亮度变化，以及随距离衰减的亮度梯度。

这些增强都不改变几何形状，所以标注框不用动。
"""
from __future__ import annotations

import numpy as np


def speckle_noise(img: np.ndarray, rng: np.random.Generator, looks: float = 4.0) -> np.ndarray:
    """乘以均值为 1 的伽马分布噪声。looks 越小噪声越强（looks=1 是完全发育的斑点）。"""
    x = img.astype(np.float32)
    noise = rng.gamma(shape=looks, scale=1.0 / looks, size=x.shape).astype(np.float32)
    return np.clip(x * noise, 0, 255).astype(np.uint8)


def random_gain(img: np.ndarray, rng: np.random.Generator, low: float = 0.6, high: float = 1.4) -> np.ndarray:
    """整体增益：模拟操作员调增益。"""
    g = rng.uniform(low, high)
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def range_gain_ramp(img: np.ndarray, rng: np.random.Generator, max_slope: float = 0.5) -> np.ndarray:
    """沿距离方向（行）叠加一个线性增益斜坡：模拟时变增益（TVG）设置不同。"""
    h = img.shape[0]
    slope = rng.uniform(-max_slope, max_slope)
    ramp = 1.0 + slope * (np.linspace(0, 1, h, dtype=np.float32) - 0.5)
    return np.clip(img.astype(np.float32) * ramp[:, None], 0, 255).astype(np.uint8)


def beam_line(img: np.ndarray, rng: np.random.Generator, strength: float = 60.0) -> np.ndarray:
    """在随机一列叠加一条亮线：模拟旁瓣泄漏、0° 亮线这类干扰，让模型学会不把它当目标。"""
    out = img.astype(np.float32).copy()
    col = rng.integers(0, img.shape[1])
    width = int(rng.integers(1, 4))
    out[:, col:col + width] += strength * rng.uniform(0.5, 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def sonar_augment(img: np.ndarray, rng: np.random.Generator,
                  p_speckle: float = 0.8, p_gain: float = 0.8,
                  p_ramp: float = 0.5, p_line: float = 0.2) -> np.ndarray:
    """组合增强。输入输出都是 uint8 单通道。"""
    out = img
    if rng.random() < p_gain:
        out = random_gain(out, rng)
    if rng.random() < p_ramp:
        out = range_gain_ramp(out, rng)
    if rng.random() < p_speckle:
        out = speckle_noise(out, rng, looks=float(rng.uniform(2.0, 8.0)))
    if rng.random() < p_line:
        out = beam_line(out, rng)
    return out
