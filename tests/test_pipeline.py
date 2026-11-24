"""用合成数据检查整条流程。先运行：python scripts/make_synthetic.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sonar.augment import sonar_augment  # noqa: E402
from sonar.breakdown import match_dataset, recall_table  # noqa: E402
from sonar.convert import write_data_yaml, write_split, yolo_lines  # noqa: E402
from sonar.geometry import polar_to_fan  # noqa: E402
from sonar.io import read_gray  # noqa: E402
from sonar.preprocess import PIPELINES, sonar_preprocess  # noqa: E402
from sonar.split import block_split, leakage_report, random_split  # noqa: E402
from sonar.stats import boxes_table  # noqa: E402
from sonar.uatd import collect_classes, find_pairs  # noqa: E402

SYN = ROOT / "data" / "synthetic" / "UATD_Training"


@pytest.fixture(scope="module")
def anns():
    if not SYN.exists():
        pytest.skip("先运行 python scripts/make_synthetic.py")
    return find_pairs(SYN)


def test_parse(anns):
    assert len(anns) > 0
    a = anns[0]
    assert a.width > 0 and a.height > 0
    assert "sonar.range_m" in a.meta
    assert all(b.x2 > b.x1 and b.y2 > b.y1 for x in anns for b in x.boxes)
    # 自然排序：seq00_0002 在 seq00_0010 之前
    stems = [x.stem for x in anns[:15]]
    assert stems == sorted(stems, key=lambda s: int(s.split("_")[1]) + 10000 * int(s[3:5]))


def test_yolo_roundtrip(anns):
    a = next(x for x in anns if x.boxes)
    classes = collect_classes(anns)
    for line, b in zip(yolo_lines(a, classes, a.width, a.height), a.boxes):
        k, cx, cy, w, h = map(float, line.split())
        assert classes[int(k)] == b.cls
        assert abs(cx * a.width - b.cx) < 0.01 and abs(h * a.height - b.h) < 0.01


def test_block_split_less_leaky(anns):
    tr_b, va_b = block_split(anns, block_size=25)
    tr_r, va_r = random_split(anns)
    assert not {a.stem for a in tr_b} & {a.stem for a in va_b}
    leak_b = leakage_report(tr_b, va_b)["mean_best_similarity"]
    leak_r = leakage_report(tr_r, va_r)["mean_best_similarity"]
    assert leak_r > leak_b + 0.2, (leak_r, leak_b)


@pytest.mark.parametrize("mode", list(PIPELINES))
def test_preprocess(anns, mode):
    img = read_gray(anns[0].image_path)
    out = sonar_preprocess(img, mode)
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_preprocess_lifts_dark_pixels(anns):
    img = read_gray(anns[0].image_path)
    assert np.median(sonar_preprocess(img, "log_clahe")) > np.median(sonar_preprocess(img, "raw"))


def test_augment_keeps_shape(anns):
    img = sonar_preprocess(read_gray(anns[0].image_path), "log_clahe")
    rng = np.random.default_rng(0)
    out = sonar_augment(img, rng, p_speckle=1, p_gain=1, p_ramp=1, p_line=1)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_fan(anns):
    img = read_gray(anns[0].image_path)
    fan = polar_to_fan(img, fov_deg=120)
    assert fan.shape[0] == img.shape[0] and fan.shape[1] > 0
    assert fan[0, 0] == 0  # 左上角在扇形外


def test_stats(anns):
    df = boxes_table(anns)
    assert {"size_bin", "range_bin"} <= set(df.columns)
    assert df["range_frac"].between(0, 1).all()


def test_convert_and_breakdown(anns, tmp_path):
    classes = collect_classes(anns)
    sub = anns[:10]
    write_split(sub, "test", tmp_path, classes)
    write_data_yaml(tmp_path, classes, has_test=True)
    gt_dir = tmp_path / "labels" / "test"
    sizes = {a.stem: (a.width, a.height) for a in sub}

    # 完美预测：真值加上置信度
    pred = tmp_path / "pred"
    pred.mkdir()
    for p in gt_dir.glob("*.txt"):
        lines = [ln + " 0.9" for ln in p.read_text().splitlines() if ln.strip()]
        (pred / p.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    gt_df, fp_df = match_dataset(gt_dir, pred, sizes)
    assert gt_df["detected"].all() and len(fp_df) == 0
    assert recall_table(gt_df, "size_bin")["recall"].dropna().eq(1.0).all()

    # 全部预测成错误类别：召回为 0，每个预测都是误报
    shutil.rmtree(pred)
    pred.mkdir()
    n_pred = 0
    for p in gt_dir.glob("*.txt"):
        lines = []
        for ln in p.read_text().splitlines():
            k, *rest = ln.split()
            lines.append(" ".join([str((int(k) + 1) % len(classes)), *rest, "0.9"]))
        n_pred += len(lines)
        (pred / p.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    gt_df, fp_df = match_dataset(gt_dir, pred, sizes)
    assert not gt_df["detected"].any() and len(fp_df) == n_pred
