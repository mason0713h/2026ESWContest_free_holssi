"""
scripts/generate_report_figures.py
===================================
서류 제출용 그래프 5종을 생성한다. 실제 IWR6843 레이더/Jetson Nano 보드
없이도 측정 가능한 항목만 다룬다 (F1/Precision/Recall, TensorRT 속도
비교처럼 실제 라벨링 데이터/실보드가 필요한 항목은 포함하지 않음 —
허위 수치가 되는 것을 피하기 위함).

생성 그래프:
    1. fig1_pipeline_latency.png   - 파이프라인 단계별 처리 시간 분포
    2. fig2_dbscan_noise_ratio.png - DBSCAN 노이즈 제거 비율 분포
    3. fig3_longlie_timeline.png   - Long-lie 알림 단계 전이 타임라인
    4. fig4_model_params.png       - 모델 파라미터 구성 (모듈별)
    5. fig5_onnx_size.png          - PyTorch(.pt) vs ONNX(.onnx) 모델 크기 비교

사용:
    python scripts/generate_report_figures.py
    (결과는 reports/figures/ 에 저장됨)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.disable(logging.CRITICAL)  # 그래프 생성 중 노이즈 로그 끄기

OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False
# 한글 폰트가 없는 환경(이 샌드박스 포함)에서는 한글이 깨질 수 있으므로
# 그래프 라벨은 영문으로 작성한다. 실제 보드/로컬 환경에 한글 폰트
# (예: NanumGothic)가 있다면 아래 주석을 해제해 한글로 바꿔도 된다.
# plt.rcParams["font.family"] = "NanumGothic"


def fig1_pipeline_latency(n_windows: int = 60) -> None:
    """파이프라인 단계별(전처리/트래킹/추론) 처리 시간을 측정해 box plot으로 그린다."""
    from radar.capture import RadarCapture
    from radar.preprocessor import PointCloudPreprocessor
    from radar.multi_tracker import MultiPersonTracker
    from model.fall_detector import FallDetector
    import torch

    capture = RadarCapture(mock_mode=True, window_size=16, stride=8)
    preprocessor = PointCloudPreprocessor(target_points=64)
    tracker = MultiPersonTracker()
    model = FallDetector()
    model.eval()

    stage_times = {"preprocess": [], "tracking": [], "inference": []}

    window = []
    for i in range(n_windows + 16):
        pc = capture._generate_mock_frame(scenario="normal")
        window.append(pc)
        if len(window) > 16:
            window.pop(0)
        if len(window) < 16:
            continue

        t0 = time.perf_counter()
        tensor = preprocessor.process_window(window)
        t1 = time.perf_counter()

        tracker.update(window[-1].points)
        t2 = time.perf_counter()

        with torch.no_grad():
            x = torch.from_numpy(tensor).unsqueeze(0).float()
            _ = model.predict_proba(x)
        t3 = time.perf_counter()

        stage_times["preprocess"].append((t1 - t0) * 1000)
        stage_times["tracking"].append((t2 - t1) * 1000)
        stage_times["inference"].append((t3 - t2) * 1000)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Preprocess\n(DBSCAN+norm)", "Tracking\n(Kalman)", "Inference\n(PyTorch CPU, untrained)"]
    data = [stage_times["preprocess"], stage_times["tracking"], stage_times["inference"]]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#55A868", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        f"Pipeline Stage Latency (n={len(stage_times['preprocess'])} windows)\n"
        "Measured on dev sandbox CPU, NOT on Jetson Nano / TensorRT"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_pipeline_latency.png", dpi=150)
    plt.close(fig)
    print(f"[1/5] fig1_pipeline_latency.png  (preprocess median={np.median(stage_times['preprocess']):.2f}ms, "
          f"tracking median={np.median(stage_times['tracking']):.2f}ms, "
          f"inference median={np.median(stage_times['inference']):.2f}ms)")


def fig2_dbscan_noise_ratio(n_frames: int = 300) -> None:
    """다양한 시나리오에서 DBSCAN 노이즈 제거 비율 분포를 그린다."""
    from radar.capture import RadarCapture
    from radar.preprocessor import PointCloudPreprocessor

    capture = RadarCapture(mock_mode=True)
    preprocessor = PointCloudPreprocessor()

    ratios = {"normal": [], "fall": []}
    for scenario in ratios:
        for _ in range(n_frames):
            pc = capture._generate_mock_frame(scenario=scenario)
            n_before = len(pc.points)
            if n_before == 0:
                continue
            denoised = preprocessor.denoise(pc.points)
            removed_ratio = 1.0 - (len(denoised) / n_before)
            ratios[scenario].append(removed_ratio * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [ratios["normal"], ratios["fall"]],
        tick_labels=["Standing (normal)", "Fallen (on floor)"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Noise points removed by DBSCAN (%)")
    ax.set_title(f"DBSCAN Denoising Ratio by Scenario (mock point cloud, n={n_frames}/scenario)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_dbscan_noise_ratio.png", dpi=150)
    plt.close(fig)
    print(f"[2/5] fig2_dbscan_noise_ratio.png  (normal median={np.median(ratios['normal']):.1f}%, "
          f"fall median={np.median(ratios['fall']):.1f}%)")


def fig3_longlie_timeline() -> None:
    """가상 낙상 시나리오를 시간축으로 가속해 Long-lie 알림 단계 전이를 그린다."""
    from model.long_lie_detector import LongLieDetector, AlertLevel
    from radar.capture import RadarCapture

    capture = RadarCapture(mock_mode=True)
    detector = LongLieDetector(
        z_floor_threshold=0.3, level1_seconds=30, level2_seconds=60, level3_seconds=120
    )

    fake_now = [0.0]

    def fake_time():
        return fake_now[0]

    timeline_t, timeline_level = [], []
    with mock.patch("model.long_lie_detector.time.time", side_effect=fake_time):
        # 0~150초: 낙상 후 계속 바닥에 누워있는 시나리오 (자력 회복 없음)
        for t in range(0, 151, 2):
            fake_now[0] = float(t)
            pc = capture._generate_mock_frame(scenario="fall")
            level = detector.update(
                person_id=1, fall_detected=True, fall_confidence=0.93, person_points=pc.points
            )
            timeline_t.append(t)
            timeline_level.append(int(level))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(timeline_t, timeline_level, where="post", linewidth=2, color="#C44E52")
    level_names = [lvl.name for lvl in AlertLevel]
    ax.set_yticks(range(len(level_names)))
    ax.set_yticklabels(level_names)
    ax.set_xlabel("Time since fall detected (s)")
    ax.set_title("Long-lie Alert Level Escalation Timeline (simulated, no self-recovery)")
    for boundary, label in [(30, "L2: 30s"), (60, "L3: 60s"), (120, "L4: 120s (emergency)")]:
        ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
        ax.text(boundary + 1, 0.3, label, rotation=90, fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_longlie_timeline.png", dpi=150)
    plt.close(fig)
    print("[3/5] fig3_longlie_timeline.png")


def fig4_model_params() -> None:
    """모델 모듈별 파라미터 수 구성을 bar chart로 그린다."""
    from model.fall_detector import FallDetector

    model = FallDetector()

    def count(m):
        return sum(p.numel() for p in m.parameters())

    pn = model.td_pointnet.pointnet
    groups = {
        "T-Net (input)": count(pn.tnet_input) if pn.use_tnet_input else 0,
        "T-Net (feature)": count(pn.tnet_feat) if pn.use_tnet_feature else 0,
        "PointNet conv/BN": (
            count(pn) - (count(pn.tnet_input) if pn.use_tnet_input else 0)
            - (count(pn.tnet_feat) if pn.use_tnet_feature else 0)
        ),
        "GRU": count(model.gru_classifier.gru),
        "Classifier MLP": count(model.gru_classifier.mlp),
    }
    total = sum(groups.values())
    assert abs(total - count(model)) <= 8  # BatchNorm running stats 등 미세한 차이만 허용

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(groups.keys())
    values = [v / 1e6 for v in groups.values()]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}M", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Parameters (Millions)")
    ax.set_title(f"FallDetector Parameter Breakdown (total={total/1e6:.2f}M)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_model_params.png", dpi=150)
    plt.close(fig)
    print(f"[4/5] fig4_model_params.png  (total={total/1e6:.2f}M params)")


def fig5_onnx_size() -> None:
    """PyTorch(.pt) state_dict vs ONNX(.onnx) export 후 파일 크기를 비교한다."""
    import torch
    from model.fall_detector import FallDetector

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("[5/5] onnx 패키지 미설치 - fig5 스킵 (pip install onnx 필요)")
        return

    model = FallDetector()
    model.eval()

    tmp_dir = OUT_DIR / "_tmp_export"
    tmp_dir.mkdir(exist_ok=True)
    pt_path = tmp_dir / "fall_detector.pt"
    onnx_path = tmp_dir / "fall_detector.onnx"

    torch.save(model.state_dict(), pt_path)

    dummy = torch.randn(1, 16, 64, 6)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["points"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )

    pt_size = pt_path.stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(["PyTorch (.pt)\nstate_dict", "ONNX (.onnx)\nopset13"], [pt_size, onnx_size],
                   color=["#4C72B0", "#55A868"])
    for bar, v in zip(bars, [pt_size, onnx_size]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f} MB", ha="center", va="bottom")
    ax.set_ylabel("File size (MB)")
    ax.set_title("Model Size: PyTorch checkpoint vs ONNX export\n(untrained weights, FP32, before TensorRT FP16/INT8 quantization)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_onnx_size.png", dpi=150)
    plt.close(fig)

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    print(f"[5/5] fig5_onnx_size.png  (.pt={pt_size:.2f}MB, .onnx={onnx_size:.2f}MB)")


if __name__ == "__main__":
    print(f"출력 디렉토리: {OUT_DIR}")
    fig1_pipeline_latency()
    fig2_dbscan_noise_ratio()
    fig3_longlie_timeline()
    fig4_model_params()
    fig5_onnx_size()
    print("\n완료. 5개 그래프가 reports/figures/ 에 저장되었습니다.")
    print("주의: 모두 mock 데이터 / 학습되지 않은 랜덤 가중치 기준이며, ")
    print("실제 성능(F1 등)이나 TensorRT 가속 효과를 나타내지 않습니다.")
