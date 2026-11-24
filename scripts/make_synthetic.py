"""生成一套「仿 UATD 格式」的合成声呐数据，用来在下载真实数据之前把整条流程跑通。

注意：这是为了测试代码而做的粗糙模拟，图像统计特性和真实声呐差别很大，
不要用它训练出来的分数说明任何问题。

模拟了真实声呐的几个关键现象：
- 极坐标矩阵：行 = 距离（上近下远），列 = 波束；
- 乘性斑点噪声 + 随距离衰减的背景，整体很暗；
- 目标是亮斑，后方（更远处）拖一条声影；
- 会动的「鱼群」小亮点、偶尔出现的波束亮线（干扰，不标注）；
- 连续帧：同一序列里目标位置缓慢漂移，用来演示随机划分的泄漏。

用法：python scripts/make_synthetic.py --out data/synthetic
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonar.io import write_image  # noqa: E402

# 类别 → (宽, 高) 像素范围
CLASS_SIZES = {
    "cube": (10, 18), "ball": (8, 14), "cylinder": (12, 26),
    "tyre": (14, 24), "human body": (8, 30),
}


def _background(rng, h, w):
    base = 18 * np.exp(-np.linspace(0, 2.0, h))[:, None] + 3          # 随距离衰减
    seabed = np.zeros((h, 1), np.float32)
    start = int(rng.uniform(0.25, 0.5) * h)
    seabed[start:] = 10 * np.exp(-np.linspace(0, 1.5, h - start))[:, None]
    img = (base + seabed) * np.ones((1, w), np.float32)
    return img


def _render(rng, h, w, objects, fish, line_col):
    img = _background(rng, h, w)
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        # 声影：目标后方（更远）同一组波束变暗
        shadow_len = int((y2 - y1) * rng.uniform(1.5, 3.0))
        img[int(y2):min(h, int(y2) + shadow_len), int(x1):int(x2)] *= 0.15
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
        cv2.ellipse(img, (int(cx), int(cy)), axes, 0, 0, 360, float(obj["strength"]), -1)
        # 目标正面（朝声呐的一侧，近端）回波更强
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y1)), float(min(255, obj["strength"] * 1.3)), 2)
    for fx, fy in fish:
        cv2.circle(img, (int(fx), int(fy)), int(rng.integers(1, 3)), float(rng.uniform(60, 140)), -1)
    if line_col is not None:
        img[:, line_col:line_col + 2] += 45
    speckle = rng.gamma(shape=1.5, scale=1 / 1.5, size=img.shape).astype(np.float32)
    img = img * speckle
    img = cv2.GaussianBlur(img, (3, 1), 0)                           # 方位方向略模糊
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_xml(path: Path, stem: str, w: int, h: int, objects, range_m: float):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = f"{stem}.bmp"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "1"
    sonar = ET.SubElement(root, "sonar")
    ET.SubElement(sonar, "range_m").text = f"{range_m:.1f}"
    ET.SubElement(sonar, "fov_deg").text = "120"
    ET.SubElement(sonar, "note").text = "synthetic"
    for obj in objects:
        o = ET.SubElement(root, "object")
        ET.SubElement(o, "name").text = obj["cls"]
        bb = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), obj["box"]):
            ET.SubElement(bb, k).text = str(int(round(v)))
    ET.ElementTree(root).write(path, encoding="utf-8")


def make_split(out: Path, n_seq: int, n_frames: int, seed: int, h=400, w=320):
    rng = np.random.default_rng(seed)
    classes = list(CLASS_SIZES)
    for s in range(n_seq):
        range_m = float(rng.choice([20, 30, 50]))
        objs = []
        for _ in range(int(rng.integers(1, 4))):
            cls = classes[rng.integers(len(classes))]
            lo, hi = CLASS_SIZES[cls]
            bw, bh = rng.uniform(lo, hi), rng.uniform(lo, hi) * 0.7
            cx, cy = rng.uniform(30, w - 30), rng.uniform(40, h - 60)
            objs.append({"cls": cls, "cx": cx, "cy": cy, "bw": bw, "bh": bh,
                         "vx": rng.normal(0, 0.6), "vy": rng.normal(-0.8, 0.4),
                         "strength": rng.uniform(90, 255)})
        fish = rng.uniform([0, 0], [w, h], size=(int(rng.integers(0, 25)), 2))
        for f in range(n_frames):
            frame_objs = []
            for o in objs:
                cx, cy = o["cx"] + o["vx"] * f, o["cy"] + o["vy"] * f
                box = [cx - o["bw"] / 2, cy - o["bh"] / 2, cx + o["bw"] / 2, cy + o["bh"] / 2]
                if box[0] < 0 or box[1] < 0 or box[2] >= w or box[3] >= h:
                    continue
                frame_objs.append({"cls": o["cls"], "box": box, "strength": o["strength"]})
            fish = (fish + rng.normal(0, 3, size=fish.shape)) % [w, h]
            line_col = int(w / 2) if rng.random() < 0.3 else None
            stem = f"seq{s:02d}_{f:04d}"
            img = _render(rng, h, w, frame_objs, fish, line_col)
            write_image(out / "image" / f"{stem}.bmp", img)
            (out / "annotation").mkdir(parents=True, exist_ok=True)
            _write_xml(out / "annotation" / f"{stem}.xml", stem, w, h, frame_objs, range_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--train-seq", type=int, default=12)
    ap.add_argument("--test-seq", type=int, default=3)
    ap.add_argument("--frames", type=int, default=25)
    args = ap.parse_args()
    out = Path(args.out)
    make_split(out / "UATD_Training", args.train_seq, args.frames, seed=1)
    make_split(out / "UATD_Test_1", args.test_seq, args.frames, seed=2)
    print(f"合成数据已写入 {out.resolve()}")


if __name__ == "__main__":
    main()
