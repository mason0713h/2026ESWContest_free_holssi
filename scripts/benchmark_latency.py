"""
scripts/benchmark_latency.py
============================
추론 지연시간 벤치마크 스크립트.

TensorRT / PyTorch / ONNX Runtime 추론 엔진별 지연시간을 측정하고
Jetson Nano(4GB, JetPack 4.6.x)에서의 실시간성 (< 50ms 목표)을 검증한다.
단, 이 스크립트 자체는 CPU(PyTorch eager)/TRT 폴백 경로로 동작하므로
샌드박스/개발 PC에서 측정한 값은 참고용일 뿐이며, 실제 Jetson Nano
TensorRT FP16 엔진 빌드 후 재측정이 필요하다.

측정 항목:
  - 전처리 (DBSCAN + 정규화) 지연시간
  - PointNet 추론 지연시간
  - GRU 추론 지연시간
  - 전체 파이프라인 엔드-투-엔드 지연시간
  - TRT / PyTorch / ONNX 비교

사용법:
    python scripts/benchmark_latency.py --n_iter 100
    python scripts/benchmark_latency.py --warmup 50 --n_iter 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_fn(fn, n_iter: int = 100, warmup: int = 20) -> Dict:
    """
    함수 지연시간 벤치마크.

    Args:
        fn: 벤치마크할 callable
        n_iter: 측정 반복 횟수
        warmup: 워밍업 반복 횟수

    Returns:
        통계 딕셔너리
    """
    # 워밍업
    for _ in range(warmup):
        fn()

    latencies = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n_iter": n_iter,
    }


def print_result(name: str, stats: Dict, target_ms: float = 50.0) -> None:
    """결과 출력."""
    ok = "✓" if stats["p95_ms"] < target_ms else "✗"
    print(f"\n[{ok}] {name}")
    print(f"     평균: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms")
    print(f"     최소: {stats['min_ms']:.2f}ms | 최대: {stats['max_ms']:.2f}ms")
    print(f"     P50: {stats['p50_ms']:.2f}ms | P95: {stats['p95_ms']:.2f}ms | P99: {stats['p99_ms']:.2f}ms")
    if stats["p95_ms"] >= target_ms:
        print(f"     ⚠ P95 초과 (목표: {target_ms}ms)")


def run_benchmarks(n_iter: int = 100, warmup: int = 20) -> None:
    """전체 벤치마크 실행."""
    print("=" * 60)
    print("  Fall Guardian 추론 지연시간 벤치마크")
    print(f"  반복: {n_iter}회, 워밍업: {warmup}회")
    print("=" * 60)

    T, N, C = 16, 64, 6

    # ─── 1. 전처리 벤치마크 ────────────────────
    print("\n[전처리 벤치마크]")
    from radar.capture import RadarCapture
    from radar.preprocessor import PointCloudPreprocessor

    capture = RadarCapture(mock_mode=True, window_size=T, stride=T)
    preprocessor = PointCloudPreprocessor(target_points=N)

    window = [capture._generate_mock_frame("normal") for _ in range(T)]

    def bench_preprocess():
        preprocessor.process_window(window)

    stats = benchmark_fn(bench_preprocess, n_iter=n_iter, warmup=warmup)
    print_result("전처리 (DBSCAN + 정규화)", stats, target_ms=20.0)

    # ─── 2. PointNet 벤치마크 ─────────────────
    print("\n[PointNet 벤치마크]")
    try:
        import torch
        from model.pointnet import TimeDistributedPointNet

        td_pn = TimeDistributedPointNet(input_dim=C, output_dim=1024)
        td_pn.eval()
        x_pt = torch.randn(1, T, N, C)

        def bench_pointnet():
            with torch.no_grad():
                td_pn(x_pt)

        stats = benchmark_fn(bench_pointnet, n_iter=n_iter, warmup=warmup)
        print_result("PointNet (CPU)", stats, target_ms=30.0)

        if torch.cuda.is_available():
            td_pn_gpu = TimeDistributedPointNet(input_dim=C, output_dim=1024).cuda()
            td_pn_gpu.eval()
            x_gpu = torch.randn(1, T, N, C).cuda()

            def bench_pointnet_gpu():
                with torch.no_grad():
                    td_pn_gpu(x_gpu)
                torch.cuda.synchronize()

            stats_gpu = benchmark_fn(bench_pointnet_gpu, n_iter=n_iter, warmup=warmup)
            print_result("PointNet (CUDA)", stats_gpu, target_ms=15.0)
    except ImportError as e:
        print(f"  PyTorch 없음: {e}")

    # ─── 3. GRU 분류기 벤치마크 ──────────────
    print("\n[GRU 분류기 벤치마크]")
    try:
        import torch
        from model.gru_classifier import GRUClassifier

        gru = GRUClassifier(input_dim=1024, hidden_size=256, num_layers=2)
        gru.eval()
        feat = torch.randn(1, T, 1024)

        def bench_gru():
            with torch.no_grad():
                gru(feat)

        stats = benchmark_fn(bench_gru, n_iter=n_iter, warmup=warmup)
        print_result("GRU 분류기 (CPU)", stats, target_ms=10.0)
    except ImportError:
        pass

    # ─── 4. 전체 파이프라인 벤치마크 ──────────
    print("\n[전체 파이프라인 벤치마크]")
    try:
        import torch
        from model.fall_detector import FallDetector

        model = FallDetector()
        model.eval()
        x_full = torch.randn(1, T, N, C)

        def bench_full():
            tensor = preprocessor.process_window(window)
            x_t = torch.from_numpy(tensor).unsqueeze(0)
            with torch.no_grad():
                model(x_t)

        stats = benchmark_fn(bench_full, n_iter=n_iter, warmup=warmup)
        print_result("전체 파이프라인 (CPU)", stats, target_ms=50.0)
    except Exception as e:
        print(f"  전체 파이프라인 오류: {e}")

    # ─── 5. TRT 엔진 벤치마크 ────────────────
    print("\n[TensorRT 엔진 벤치마크]")
    from model.tensorrt_inference import TRTInferenceEngine, TRT_AVAILABLE

    if TRT_AVAILABLE:
        engine = TRTInferenceEngine("model/weights/fall_guardian.trt")
        if engine.load():
            x_np = np.random.randn(1, T, N, C).astype(np.float32)

            def bench_trt():
                engine.infer(x_np)

            stats = benchmark_fn(bench_trt, n_iter=n_iter, warmup=warmup)
            print_result("TensorRT FP16", stats, target_ms=50.0)
            engine.release()
        else:
            print("  TRT 엔진 파일 없음. 빌드 후 재시도하세요.")
    else:
        print("  TensorRT 미설치. Fallback 모드 벤치마크:")
        engine = TRTInferenceEngine("model/weights/fall_guardian.trt")
        x_np = np.random.randn(1, T, N, C).astype(np.float32)

        def bench_trt_fallback():
            engine.infer(x_np)

        stats = benchmark_fn(bench_trt_fallback, n_iter=n_iter, warmup=warmup)
        print_result("TRT Fallback (Mock)", stats, target_ms=50.0)

    # ─── 요약 ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  벤치마크 완료")
    print("  목표: 전체 파이프라인 P95 < 50ms (레이더 프레임 주기: 100ms)")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Fall Guardian 추론 벤치마크")
    parser.add_argument("--n_iter", type=int, default=100, help="측정 반복 횟수")
    parser.add_argument("--warmup", type=int, default=20, help="워밍업 반복 횟수")
    args = parser.parse_args()

    run_benchmarks(n_iter=args.n_iter, warmup=args.warmup)
