"""
training/evaluate.py
====================
낙상 감지 모델 평가 스크립트.

메트릭:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
  - Confusion Matrix (시각화 포함)
  - 오경보율 (False Alarm Rate)
  - 평균 추론 지연시간
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.fall_detector import FallDetector
from training.dataset import FallDataset, create_dataloaders

logger = logging.getLogger(__name__)


def evaluate_model(
    model: FallDetector,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.7,
) -> Dict:
    """
    모델 전체 평가.

    Args:
        model: 평가할 FallDetector
        dataloader: 평가 DataLoader
        device: 디바이스
        threshold: 낙상 판정 임계값

    Returns:
        메트릭 딕셔너리
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probas = []
    all_latencies = []

    import time

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            t0 = time.perf_counter()
            logits, _ = model(x)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(y)  # 샘플당
            all_latencies.append(elapsed_ms)

            proba = F.softmax(logits, dim=-1)[:, 1]  # 낙상 확률
            pred = (proba >= threshold).long()

            all_labels.extend(y.cpu().numpy().tolist())
            all_preds.extend(pred.cpu().numpy().tolist())
            all_probas.extend(proba.cpu().numpy().tolist())

    return compute_metrics(all_labels, all_preds, all_probas, all_latencies)


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probas: Optional[List[float]] = None,
    latencies: Optional[List[float]] = None,
) -> Dict:
    """
    분류 메트릭 계산.

    Args:
        labels: 정답 레이블 리스트
        preds: 예측 레이블 리스트
        probas: 낙상 확률 리스트 (ROC-AUC 계산용)
        latencies: 추론 지연시간 리스트 (ms)

    Returns:
        전체 메트릭 딕셔너리
    """
    labels = np.array(labels)
    preds = np.array(preds)

    # Confusion Matrix
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))

    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_samples": total,
        "fall_samples": int(np.sum(labels == 1)),
        "normal_samples": int(np.sum(labels == 0)),
    }

    # ROC-AUC
    if probas is not None:
        try:
            auc = _compute_auc(labels, np.array(probas))
            metrics["roc_auc"] = auc
        except Exception as e:
            logger.debug("AUC 계산 오류: %s", e)
            metrics["roc_auc"] = 0.0

    # 지연시간 통계
    if latencies is not None:
        lat_arr = np.array(latencies)
        metrics["latency_mean_ms"] = float(lat_arr.mean())
        metrics["latency_p95_ms"] = float(np.percentile(lat_arr, 95))
        metrics["latency_max_ms"] = float(lat_arr.max())

    return metrics


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    트래피조이드 룰로 ROC-AUC 계산 (sklearn 없이).

    Args:
        labels: 이진 레이블 (0/1)
        scores: 낙상 확률 점수

    Returns:
        AUC 값 (0~1)
    """
    thresholds = np.unique(scores)[::-1]
    tpr_list = []
    fpr_list = []

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    for th in thresholds:
        preds = (scores >= th).astype(int)
        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # 정렬
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in sorted_pairs]
    tpr_sorted = [p[1] for p in sorted_pairs]

    # 트래피조이드 적분
    auc = float(np.trapz(tpr_sorted, fpr_sorted))
    return abs(auc)


def print_report(metrics: Dict, title: str = "평가 결과") -> None:
    """메트릭 보고서 출력."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  전체 샘플: {metrics['total_samples']} "
          f"(낙상={metrics['fall_samples']}, 정상={metrics['normal_samples']})")
    print(f"{'─'*60}")
    print(f"  {'Accuracy':<20}: {metrics['accuracy']:.4f}")
    print(f"  {'Precision':<20}: {metrics['precision']:.4f}")
    print(f"  {'Recall (민감도)':<20}: {metrics['recall']:.4f}")
    print(f"  {'F1-Score':<20}: {metrics['f1']:.4f}")
    print(f"  {'Specificity':<20}: {metrics['specificity']:.4f}")
    print(f"  {'오경보율 (FAR)':<20}: {metrics['false_alarm_rate']:.4f}")
    print(f"  {'미탐지율':<20}: {metrics['miss_rate']:.4f}")

    if "roc_auc" in metrics:
        print(f"  {'ROC-AUC':<20}: {metrics['roc_auc']:.4f}")

    print(f"{'─'*60}")
    print(f"  Confusion Matrix:")
    print(f"         Pred_Normal  Pred_Fall")
    print(f"  True_N     {metrics['tn']:6d}       {metrics['fp']:6d}")
    print(f"  True_F     {metrics['fn']:6d}       {metrics['tp']:6d}")

    if "latency_mean_ms" in metrics:
        print(f"{'─'*60}")
        print(f"  추론 지연시간 (샘플당):")
        print(f"    평균: {metrics['latency_mean_ms']:.2f}ms")
        print(f"    P95:  {metrics['latency_p95_ms']:.2f}ms")
        print(f"    최대: {metrics['latency_max_ms']:.2f}ms")

    print(f"{'='*60}\n")


def save_confusion_matrix(
    metrics: Dict,
    output_path: str,
) -> None:
    """Confusion Matrix 이미지 저장 (matplotlib 사용 시)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["정상", "낙상"], fontsize=12)
        ax.set_yticklabels(["정상", "낙상"], fontsize=12)
        ax.set_xlabel("예측", fontsize=12)
        ax.set_ylabel("실제", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Confusion Matrix 저장: %s", output_path)
    except ImportError:
        logger.warning("matplotlib 미설치. Confusion Matrix 이미지 저장 스킵")


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fall Guardian 모델 평가")
    parser.add_argument("--checkpoint", type=str, default="model/weights/best_checkpoint.pth")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mock_data", action="store_true")
    parser.add_argument("--save_cm", type=str, default="logs/confusion_matrix.png")
    args = parser.parse_args()

    print("=== Fall Guardian 모델 평가 ===\n")

    # 디바이스
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"평가 디바이스: {device}")

    # Mock 데이터
    data_root = args.data_root
    if args.mock_data:
        tmpdir = tempfile.mkdtemp()
        FallDataset.generate_mock_dataset(tmpdir, n_fall=50, n_normal=200)
        data_root = tmpdir

    # 데이터 로더
    _, _, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # 모델
    model = FallDetector(fall_threshold=args.threshold)

    if Path(args.checkpoint).exists():
        model.load(args.checkpoint, map_location=str(device))
        print(f"체크포인트 로드: {args.checkpoint}")
    else:
        print(f"체크포인트 없음 ({args.checkpoint}). 랜덤 가중치로 평가.")

    # 평가
    metrics = evaluate_model(model, test_loader, device, threshold=args.threshold)
    print_report(metrics, "테스트 세트 평가 결과")

    # Confusion Matrix 저장
    save_confusion_matrix(metrics, args.save_cm)
    print("평가 완료!")
