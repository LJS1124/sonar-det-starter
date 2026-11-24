"""读取 UATD（以及同格式的 URPC 声学赛道）数据：图像 + VOC 风格 XML 标注。

UATD 的 XML 除了常规的 <object>/<bndbox>，还带有声呐参数（量程、方位角等）。
不同版本字段名可能不同，所以这里不写死字段，而是把 <object> 和 <size> 以外的
所有叶子节点都收进 meta 字典，第 1 天先打印出来看看实际有哪些字段。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# UATD 论文中的 10 个类别。实际类别以 XML 里出现的名字为准（见 collect_classes）。
UATD_CLASSES = [
    "cube", "ball", "cylinder", "human body", "tyre",
    "circle cage", "square cage", "metal bucket", "plane", "rov",
]


@dataclass
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class Annotation:
    image_path: Path
    xml_path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.image_path.stem


def natural_key(s: str):
    """按自然顺序排序：frame2 排在 frame10 前面。划分连续段时要用到。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _leaf_items(elem: ET.Element, prefix: str = ""):
    for child in elem:
        tag = child.tag.strip()
        key = f"{prefix}{tag}"
        if len(child):
            yield from _leaf_items(child, prefix=f"{key}.")
        else:
            yield key, (child.text or "").strip()


def parse_xml(xml_path: Path, image_path: Path | None = None) -> Annotation:
    root = ET.parse(xml_path).getroot()
    boxes: list[Box] = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None or not name:
            continue
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append(Box(name, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    width = height = 0
    size = root.find("size")
    if size is not None:
        width = int(float(size.findtext("width") or 0))
        height = int(float(size.findtext("height") or 0))

    meta = {}
    for child in root:
        if child.tag in ("object", "size"):
            continue
        if len(child):
            meta.update(dict(_leaf_items(child, prefix=f"{child.tag}.")))
        else:
            meta[child.tag] = (child.text or "").strip()

    if image_path is None:
        image_path = xml_path.with_suffix(".bmp")
    return Annotation(Path(image_path), Path(xml_path), width, height, boxes, meta)


def find_pairs(root_dir: str | Path) -> list[Annotation]:
    """递归查找 XML，并按文件名（不含扩展名）匹配同名图像，结果按自然顺序排列。

    兼容两种目录结构：图像和 XML 放在同一目录，或者分在 image/ 与 annotation/ 两个子目录。
    """
    root_dir = Path(root_dir)
    images = {p.stem: p for p in root_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    anns, missing = [], []
    for xml in root_dir.rglob("*.xml"):
        img = images.get(xml.stem)
        if img is None:
            missing.append(xml.name)
            continue
        anns.append(parse_xml(xml, img))
    if missing:
        print(f"[find_pairs] {len(missing)} 个 XML 找不到同名图像，例如 {missing[:3]}")
    anns.sort(key=lambda a: natural_key(a.stem))
    return anns


def collect_classes(anns: list[Annotation]) -> list[str]:
    """数据里实际出现的类别。顺序：先按 UATD 论文顺序，论文里没有的追加在后面。"""
    seen = {b.cls for a in anns for b in a.boxes}
    ordered = [c for c in UATD_CLASSES if c in seen]
    ordered += sorted(seen - set(ordered))
    return ordered
