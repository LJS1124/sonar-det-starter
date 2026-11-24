"""第 1 天：数据体检。打印 XML 字段、类别分布、框大小与距离分布、图像亮度，并保存样例图。

用法：python scripts/inspect_data.py --root data/raw/UATD_Training --out reports/inspect
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonar.geometry import polar_to_fan  # noqa: E402
from sonar.io import read_gray, to_uint8  # noqa: E402
from sonar.preprocess import sonar_preprocess  # noqa: E402
from sonar.stats import boxes_table, image_table  # noqa: E402
from sonar.uatd import collect_classes, find_pairs  # noqa: E402
from sonar.viz import draw_boxes, montage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="reports/inspect")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--fov", type=float, default=120.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    anns = find_pairs(args.root)
    print(f"图像-标注对：{len(anns)}")
    if not anns:
        return
    print("\n== 第一张图的 XML 附加字段（声呐参数在这里）==")
    for k, v in anns[0].meta.items():
        print(f"  {k}: {v}")

    classes = collect_classes(anns)
    bt = boxes_table(anns)
    it = image_table(anns)
    print(f"\n== 类别（{len(classes)}）==\n{bt['cls'].value_counts().to_string()}")
    print(f"\n== 框大小（面积开方，像素）==\n{bt['sqrt_area'].describe().round(1).to_string()}")
    print(f"\n== 框大小分桶 ==\n{bt['size_bin'].value_counts(sort=False).to_string()}")
    print(f"\n== 距离分桶（近/中/远）==\n{bt['range_bin'].value_counts(sort=False).to_string()}")
    print(f"\n== 图像（抽样 {len(it)} 张）==\n{it[['height', 'width', 'mean', 'p50', 'p99', 'max', 'dark_ratio']].describe().round(2).to_string()}")
    print(f"\n位深：{it['dtype'].value_counts().to_dict()}；无目标图像比例：{(it['n_boxes'] == 0).mean():.1%}")
    bt.to_csv(out / "boxes.csv", index=False)
    it.to_csv(out / "images.csv", index=False)

    rng = np.random.default_rng(0)
    pick = [anns[i] for i in rng.choice(len(anns), min(args.n_samples, len(anns)), replace=False)]
    raw = [draw_boxes(a, classes=classes) for a in pick]
    enh = [draw_boxes(a, img=sonar_preprocess(read_gray(a.image_path), "log_clahe"), classes=classes) for a in pick]
    fan = [polar_to_fan(sonar_preprocess(read_gray(a.image_path), "log_clahe"), fov_deg=args.fov) for a in pick]
    cv2.imwrite(str(out / "samples_raw.png"), cv2.cvtColor(montage(raw), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "samples_log_clahe.png"), cv2.cvtColor(montage(enh), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "samples_fan.png"), montage([to_uint8(f) for f in fan]))
    print(f"\n报告与样例图已保存到 {out.resolve()}")


if __name__ == "__main__":
    main()
