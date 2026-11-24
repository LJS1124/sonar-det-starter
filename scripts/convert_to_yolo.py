"""第 3 天（及第 5–8 天）：转换成 YOLO 格式，按连续块划分，可选预处理和离线增强。

基线（当成光学图，只做线性缩放）：
    python scripts/convert_to_yolo.py --train-root data/raw/UATD_Training \\
        --test-roots data/raw/UATD_Test_1 data/raw/UATD_Test_2 --out data/yolo_raw --preprocess raw

第 5–6 天，换预处理：--preprocess log_clahe（或 lee_log_clahe 等，见 sonar/preprocess.py）
第 7–8 天，再加增强副本：--augment --aug-copies 1

每次转换都会输出一个泄漏检查结果。加 --split random 可以亲眼看到随机划分的泄漏有多严重。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonar.augment import sonar_augment  # noqa: E402
from sonar.convert import write_data_yaml, write_split  # noqa: E402
from sonar.preprocess import PIPELINES, sonar_preprocess  # noqa: E402
from sonar.split import block_split, leakage_report, random_split  # noqa: E402
from sonar.uatd import collect_classes, find_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--test-roots", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--preprocess", default="raw", choices=list(PIPELINES))
    ap.add_argument("--split", default="block", choices=["block", "random"])
    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true", help="给训练集额外生成声呐增强副本")
    ap.add_argument("--aug-copies", type=int, default=1)
    # 消融实验用：把某一项设为 0 就关掉它
    ap.add_argument("--p-speckle", type=float, default=0.8)
    ap.add_argument("--p-gain", type=float, default=0.8)
    ap.add_argument("--p-ramp", type=float, default=0.5)
    ap.add_argument("--p-line", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    anns = find_pairs(args.train_root)
    tests = [a for r in args.test_roots for a in find_pairs(r)]
    classes = collect_classes(anns + tests)
    if args.split == "block":
        train, val = block_split(anns, args.val_ratio, args.block_size, args.seed)
    else:
        train, val = random_split(anns, args.val_ratio, args.seed)

    leak = leakage_report(train, val)
    print(f"划分方式：{args.split}；训练 {len(train)}，验证 {len(val)}，测试 {len(tests)}")
    print(f"泄漏检查：验证图与训练集最相似图的平均相关系数 {leak['mean_best_similarity']:.3f}，"
          f"近似重复（≥{leak['threshold']}）比例 {leak['near_duplicate_ratio']:.1%}")

    tf = partial(sonar_preprocess, mode=args.preprocess)
    aug = partial(sonar_augment, p_speckle=args.p_speckle, p_gain=args.p_gain,
                  p_ramp=args.p_ramp, p_line=args.p_line) if args.augment else None
    n_tr = write_split(train, "train", out, classes, tf, aug, args.aug_copies, args.seed)
    n_va = write_split(val, "val", out, classes, tf)
    n_te = write_split(tests, "test", out, classes, tf) if tests else 0
    yaml_path = write_data_yaml(out, classes, has_test=bool(tests))

    info = {**vars(args), "classes": classes, "n_train_files": n_tr, "n_val": n_va,
            "n_test": n_te, "leakage": leak}
    (out / "conversion.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"完成：{yaml_path}")


if __name__ == "__main__":
    main()
