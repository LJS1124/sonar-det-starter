# sonar-det-starter：声呐图像目标检测入门

给做过光学 CV 的工程师用的声呐检测练手仓库，配合《声呐图像目标检测 14 天入门教程》使用。
数据集用 [UATD](https://figshare.com/articles/dataset/UATD_Dataset/21331143/3)（多波束前视声呐，9000+ 张，10 类）。

> 这份 README 也是你最终作品集 README 的模板：带「（待填）」的地方，跑完实验后换成你自己的结论和数字。

## 一句话结论（待填）

在 UATD 上，声呐预处理和声呐增强把 test mAP50 从 A 提到 B，远距离小目标召回率从 C 提到 D；RK3588 上 INT8 达到 E FPS。

## 快速开始

```bash
# 建议单独建一个虚拟环境，不要装进已有的大环境里，避免 NumPy 版本冲突
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name sonar-det --display-name "sonar-det"   # notebook 里选这个内核

# 1. 没有真实数据也能先跑通：生成合成数据 + 跑测试
python scripts/make_synthetic.py
pytest

# 2. 下载 UATD，解压到 data/raw/UATD_Training、data/raw/UATD_Test_1、data/raw/UATD_Test_2
python scripts/inspect_data.py --root data/raw/UATD_Training --out reports/inspect

# 3. 光学对照基线
python scripts/convert_to_yolo.py --train-root data/raw/UATD_Training \
    --test-roots data/raw/UATD_Test_1 data/raw/UATD_Test_2 --out data/yolo_raw --preprocess raw
python scripts/train_baseline.py --data data/yolo_raw/data.yaml --name baseline_optical
```

## 只有 CPU 怎么办

- **CPU 上能完成的**：第 1–2 天数据分析、第 3 天转换与划分、第 5–8 天生成预处理和增强数据、第 9–10 天的误差拆解（有预测结果之后）、第 11–12 天的 ONNX CPU 测速。
- **CPU 上只做冒烟测试的**：训练。用 `--device cpu --model yolo11n.pt --imgsz 320 --fraction 0.2 --epochs 5` 确认流程能跑通，分数不要当真。
- **正式训练放到免费 GPU 上**：Kaggle Notebooks（每周约 30 GPU 小时，UATD 在 Kaggle 上有镜像，不用自己上传）；或 ModelScope 魔搭 Notebook（新用户有一次性 GPU 额度，国内访问方便）。在 notebook 里 `git clone` 本仓库，按教程的命令执行即可。

## 目录

| 路径 | 对应教程 | 内容 |
| --- | --- | --- |
| `sonar/uatd.py` | 第 1 天 | 读取图像 + XML 标注，收集声呐参数字段 |
| `sonar/geometry.py` | 第 1 天 | 极坐标矩阵展开成扇形图 |
| `sonar/stats.py`、`sonar/viz.py` | 第 1–2 天 | 统计与可视化 |
| `sonar/split.py` | 第 3 天 | 连续块划分、泄漏检查 |
| `sonar/convert.py` | 第 3 天 | 转 YOLO 格式 |
| `sonar/preprocess.py` | 第 5–6 天 | 对数压缩、分位数拉伸、CLAHE、Lee 滤波 |
| `sonar/augment.py` | 第 7–8 天 | 斑点噪声、增益、距离增益斜坡、波束亮线 |
| `sonar/breakdown.py` | 第 9–10 天 | 按大小 / 距离 / 类别的召回率，每图误报 |
| `scripts/` | 全程 | 命令行入口，见各文件开头的用法说明 |
| `notebooks/01–03` | 第 1–4 天 | 数据探索、划分与转换、基线训练 |
| `tools/build_notebooks.py` | — | notebook 的源文件，改 notebook 请改这里 |

## 数据笔记（待填，第 1–2 天）

| 问题 | 你的数据 |
| --- | --- |
| 图像尺寸、位深 | |
| 行是距离还是方位？近端在哪里？ | |
| 每个像素代表多少米 | |
| 类别数，样本最少的类别 | |
| 框大小中位数，小于 16 像素的比例 | |
| 远距离目标占比 | |
| 「几乎全黑」像素占比 | |

## 实验表（待填）

每行只改一个变量。所有分数来自 `reports/experiments.csv`。

| 实验 | 预处理 | 增强 | val mAP50 | test mAP50 | test mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| baseline_optical | 线性缩放 | Ultralytics 默认 | | | |
| pre_raw_sonaraug | 线性缩放 | 关颜色增强 | | | |
| pre_log_clahe | 对数 + CLAHE | 关颜色增强 | | | |
| pre_lee_log_clahe | Lee + 对数 + CLAHE | 关颜色增强 | | | |
| aug_speckle | 最佳预处理 | + 声呐增强 | | | |

## 误差分析（待填，第 9–10 天）

按大小、按距离的召回率对比表，失败案例图（`reports/breakdown/failure_*.png`），失败原因分布。

## 部署（待填，第 11–12 天）

| 平台 | 格式 | 精度 | 预处理 ms | 推理 ms | 总 FPS | test mAP50 |
| --- | --- | --- | --- | --- | --- | --- |

## 局限

- `data/synthetic` 只用于检查代码能不能跑，图像统计特性和真实声呐差别很大，不要用它的分数说明任何问题。
- `sonar/uatd.py` 按 VOC 风格解析 XML，并把其余字段收进 `meta`。拿到真实数据后，先用 `inspect_data.py` 确认字段名和图像方向。
- 扇形展开的开角默认 120°，请以 XML 里的方位参数或设备手册为准。

## 参考

- Xie, Yang, Qiu. A Dataset with Multibeam Forward-Looking Sonar for Underwater Object Detection. Scientific Data, 2022.
- Kay et al. The Caltech Fish Counting Dataset. ECCV 2022.
- Valdenegro-Toro. Best Practices in Convolutional Networks for Forward-Looking Sonar Image Recognition. 2017.
