"""第 11–12 天：导出 ONNX，并用 ONNX Runtime 测推理速度。

    python scripts/export_and_bench.py --weights runs/<name>/weights/best.pt --imgsz 640

精度损失：导出后用 `yolo val model=best.onnx data=... split=test` 再评估一次，和 .pt 的结果对比。
TensorRT（Jetson）：`yolo export model=best.pt format=engine half=True`。
RKNN（瑞芯微 RK3588）：先导出 ONNX，再用瑞芯微官方的 rknn-toolkit2 转换和量化，
需要准备一小批校准图（从训练集里抽 100 张左右即可）。
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def bench_onnx(onnx_path: str, imgsz: int, runs: int = 100, warmup: int = 10) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
    inp = sess.get_inputs()[0]
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"provider": sess.get_providers()[0], "mean_ms": float(t.mean()),
            "p95_ms": float(np.percentile(t, 95)), "fps": float(1000 / t.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    onnx_path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=True)
    r = bench_onnx(str(onnx_path), args.imgsz)
    print(f"{onnx_path}：{r['provider']}，平均 {r['mean_ms']:.1f} ms，P95 {r['p95_ms']:.1f} ms，约 {r['fps']:.0f} FPS")


if __name__ == "__main__":
    main()
