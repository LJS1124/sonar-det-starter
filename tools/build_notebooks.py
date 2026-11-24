"""生成 notebooks/*.ipynb。改 notebook 内容请改这个文件，再运行：python tools/build_notebooks.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''import sys
from pathlib import Path
REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sonar.plotting import setup_cjk_font
plt.rcParams["figure.dpi"] = 110
setup_cjk_font()   # 服务器上没有中文字体时会提示安装方法

# 下载了真实数据就改成 data/raw/UATD_Training；没下载就先用合成数据（python scripts/make_synthetic.py）
DATA_ROOT = REPO / "data" / "raw" / "UATD_Training"
if not DATA_ROOT.exists():
    DATA_ROOT = REPO / "data" / "synthetic" / "UATD_Training"
print("使用数据：", DATA_ROOT)'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


NB01 = [
    md("""# 第 1–2 天：看懂声呐数据

目标：在写训练代码之前，用数字回答「这批数据长什么样」。每一节最后都有一个问题，把答案记进你的数据笔记。"""),
    code(SETUP),
    md("## 1. 读数据，看 XML 里有哪些声呐参数"),
    code('''from sonar.uatd import find_pairs, collect_classes
anns = find_pairs(DATA_ROOT)
classes = collect_classes(anns)
print(f"{len(anns)} 张图，{len(classes)} 个类别：{classes}")
print("\\n第一张图的附加字段：")
for k, v in anns[0].meta.items():
    print(f"  {k}: {v}")'''),
    md("> 问题：哪个字段是量程（最大探测距离）？用「量程 ÷ 图像高度」算出每个像素代表多少米。"),
    md("## 2. 类别、框大小、目标距离的分布"),
    code('''from sonar.stats import boxes_table
bt = boxes_table(anns, near_at_top=True)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
bt["cls"].value_counts().plot.barh(ax=ax[0], title="各类别框数")
bt["sqrt_area"].plot.hist(bins=40, ax=ax[1], title="框大小（面积开方，像素）")
bt["range_frac"].plot.hist(bins=20, ax=ax[2], title="目标在距离方向的位置（0=近，1=远）")
plt.tight_layout(); plt.show()
print(bt.groupby("range_bin", observed=False)["sqrt_area"].describe().round(1))'''),
    md("> 问题：小于 16 像素的框占多少？远处目标是不是更小？哪个类别样本最少？"),
    md("## 3. 为什么原图「几乎全黑」：看像素强度分布"),
    code('''from sonar.io import read_gray, to_uint8
img = read_gray(anns[0].image_path)
print("dtype:", img.dtype, " shape:", img.shape)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].hist(img.ravel(), bins=256, log=True); ax[0].set_title("像素强度直方图（纵轴对数）")
pct = np.percentile(img, [50, 90, 99, 99.9])
ax[1].bar(["P50", "P90", "P99", "P99.9"], pct); ax[1].set_title("分位数")
plt.tight_layout(); plt.show()
print("P50/P90/P99/P99.9 =", pct, " 最大值 =", img.max())'''),
    md("> 问题：中位数是最大值的百分之几？如果只做线性缩放，一半的像素会落在 0–255 的哪一段？"),
    md("## 4. 原图、增强图、扇形图对比"),
    code('''from sonar.preprocess import sonar_preprocess
from sonar.geometry import polar_to_fan
from sonar.viz import draw_boxes
a = next(x for x in anns if x.boxes)
raw = read_gray(a.image_path)
enh = sonar_preprocess(raw, "log_clahe")
fan = polar_to_fan(enh, fov_deg=120)   # 开角以你的数据为准
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
ax[0].imshow(draw_boxes(a, classes=classes)); ax[0].set_title("原图（线性缩放）")
ax[1].imshow(draw_boxes(a, img=enh, classes=classes)); ax[1].set_title("对数 + CLAHE")
ax[2].imshow(fan, cmap="gray"); ax[2].set_title("展开成扇形（声呐在底部中央）")
for x in ax: x.axis("off")
plt.tight_layout(); plt.show()'''),
    md("> 问题：原图里你能看见几个目标？增强图里呢？扇形图里远处的目标为什么变宽了？"),
    md("## 5. 找声影：穿过目标中心的单列强度剖面\n\n亮峰后面紧跟一段暗谷，暗谷所在的一侧就是「更远」的方向。"),
    code('''b = max(a.boxes, key=lambda b: b.w * b.h)
col = int(b.cx)
profile = sonar_preprocess(raw, "log")[:, col].astype(float)
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(profile, lw=0.8)
ax.axvspan(b.y1, b.y2, color="tab:orange", alpha=0.3, label=f"目标 {b.cls}")
ax.axvspan(b.y2, min(len(profile) - 1, b.y2 + 3 * b.h), color="tab:blue", alpha=0.15, label="预期的声影区")
ax.set_xlabel("行号"); ax.set_ylabel("对数强度"); ax.legend(); plt.tight_layout(); plt.show()
before = profile[max(0, int(b.y1 - 2 * b.h)):int(b.y1)].mean()
after = profile[int(b.y2):int(b.y2 + 2 * b.h)].mean()
print(f"目标前方平均强度 {before:.1f}，目标后方平均强度 {after:.1f}")
print("结论：", "行号越大越远（near_at_top=True）" if after < before else "方向可能相反，检查一下 near_at_top")'''),
    md("> 问题：换几个目标再看，结论稳定吗？"),
    md("## 6. 连续帧有多像：为第 3 天的划分做准备"),
    code('''from sonar.split import _thumb
t = np.stack([_thumb(x) for x in anns[:60]])
sim = t @ t.T
plt.figure(figsize=(4.5, 4)); plt.imshow(sim, vmin=0, vmax=1); plt.colorbar()
plt.title("前 60 帧两两相似度"); plt.show()
print("相邻帧平均相似度：", np.mean([sim[i, i + 1] for i in range(len(t) - 1)]).round(3))'''),
    md("> 问题：对角线附近的亮块有多宽？这就是第 3 天 `block_size` 至少要设多大。"),
]

NB02 = [
    md("""# 第 3 天：转换成 YOLO 格式，并且不让测试集泄漏

这一步决定了你之后所有分数可不可信。先亲眼看看随机划分会出什么问题。"""),
    code(SETUP),
    code('''from sonar.uatd import find_pairs, collect_classes
from sonar.split import random_split, block_split, leakage_report
anns = find_pairs(DATA_ROOT)
classes = collect_classes(anns)
rows = []
for name, (tr, va) in {
    "随机划分": random_split(anns, 0.2),
    "连续块划分 block=25": block_split(anns, 0.2, block_size=25),
    "连续块划分 block=50": block_split(anns, 0.2, block_size=50),
}.items():
    r = leakage_report(tr, va)
    rows.append({"划分": name, "训练": len(tr), "验证": len(va),
                 "平均最高相似度": round(r["mean_best_similarity"], 3),
                 "近似重复比例": f'{r["near_duplicate_ratio"]:.1%}'})
pd.DataFrame(rows)'''),
    md("> 问题：随机划分的「平均最高相似度」比连续块划分高多少？用随机划分算出来的验证分数会偏高还是偏低？"),
    md("## 转换（基线：只做线性缩放，相当于当成光学图）"),
    code('''from functools import partial
from sonar.convert import write_split, write_data_yaml
from sonar.preprocess import sonar_preprocess
OUT = REPO / "data" / "yolo_raw"
train, val = block_split(anns, 0.2, block_size=50)
tf = partial(sonar_preprocess, mode="raw")
write_split(train, "train", OUT, classes, tf)
write_split(val, "val", OUT, classes, tf)
print(write_data_yaml(OUT, classes, has_test=False).read_text())
# 有测试集时直接用命令行：python scripts/convert_to_yolo.py ... --test-roots ...（见教程）'''),
    md("## 反向检查：把 YOLO 标签画回图上，确认坐标没转错"),
    code('''import cv2
img_p = sorted((OUT / "images" / "val").glob("*.png"))[0]
lbl_p = OUT / "labels" / "val" / f"{img_p.stem}.txt"
im = cv2.cvtColor(cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
H, W = im.shape[:2]
for line in lbl_p.read_text().splitlines():
    k, cx, cy, w, h = map(float, line.split())
    x1, y1, x2, y2 = int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)
    cv2.rectangle(im, (x1, y1), (x2, y2), (255, 80, 80), 1)
    cv2.putText(im, classes[int(k)], (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 80), 1)
plt.figure(figsize=(6, 6)); plt.imshow(im); plt.axis("off"); plt.title(img_p.name); plt.show()'''),
    md("> 问题：框和目标对得上吗？对不上的话，最常见的原因是宽高写反或忘了归一化。"),
]

NB03 = [
    md("""# 第 3–4 天：光学对照基线

**原封不动**地用光学默认设置训练，得到的分数是之后所有改进的对照组。需要 GPU 和 `pip install ultralytics`。

建议直接用命令行跑（结果会自动追加到 `reports/experiments.csv`）：

```bash
python scripts/train_baseline.py --data data/yolo_raw/data.yaml --name baseline_optical --model yolo11s.pt
```

下面的单元格做同样的事，方便边跑边看。"""),
    code(SETUP),
    code('''from ultralytics import YOLO
DATA = REPO / "data" / "yolo_raw" / "data.yaml"
model = YOLO("yolo11s.pt")            # COCO 预训练
model.train(data=str(DATA), epochs=100, imgsz=640, batch=16, seed=0, deterministic=True,
            project=str(REPO / "runs"), name="baseline_optical", exist_ok=True)'''),
    md("## 在验证集（和测试集）上评估"),
    code('''import yaml
best = YOLO(str(REPO / "runs" / "baseline_optical" / "weights" / "best.pt"))
splits = ["val"] + (["test"] if "test" in yaml.safe_load(DATA.read_text()) else [])
res = {s: best.val(data=str(DATA), split=s, project=str(REPO / "runs"), name=f"baseline_{s}", exist_ok=True) for s in splits}
for s, m in res.items():
    print(f"{s}: mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")'''),
    md("## 各类别 AP：哪个类最差？"),
    code('''m = res[splits[-1]]
per_cls = pd.Series(m.box.maps, index=[m.names[i] for i in range(len(m.box.maps))], name="AP50-95").sort_values()
per_cls.plot.barh(figsize=(6, 3.5), title=f"各类别 AP50-95（{splits[-1]}）"); plt.tight_layout(); plt.show()
per_cls'''),
    md("""> 问题：
> 1. 验证集和测试集的分数差多少？差得多说明什么？
> 2. 最差的两个类别，在第 1 天的统计里是不是样本最少或者框最小？
> 3. 把这组数字填进 README 的实验表第一行，标注「光学默认设置」。"""),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in [("01_explore", NB01), ("02_convert_split", NB02), ("03_baseline", NB03)]:
        (out / f"{name}.ipynb").write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1))
        print("写入", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
