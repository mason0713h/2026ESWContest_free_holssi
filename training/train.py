"""
training/train.py
=================
FallDetector 학습 스크립트.

기능:
  - AdamW 옵티마이저 + CosineAnnealingLR 스케줄러
  - 체크포인트 저장 (최고 F1 기준)
  - TensorBoard 로깅
  - Early stopping
  - 혼합 정밀도 학습 (FP16, AMP)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from model.fall_detector import FallDetector
from training.dataset import FallDataset, create_dataloaders

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping 유틸리티.

    Args:
        patience: 개선 없이 허용할 최대 에폭 수
        min_delta: 개선으로 인정할 최소 변화량
        mode: 'min' (손실 최소화) 또는 'max' (F1 최대화)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best = float("-inf") if mode == "max" else float("inf")
        self._counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        메트릭 업데이트 및 정지 여부 반환.

        Returns:
            개선된 경우 True, 미개선이면 False
        """
        if self.mode == "max":
            improved = metric > self._best + self.min_delta
        else:
            improved = metric < self._best - self.min_delta

        if improved:
            self._best = metric
            self._counter = 0
            return True
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
            return False


class Trainer:
    """
    FallDetector 학습기.

    Args:
        model: 학습할 FallDetector 모델
        train_loader: 학습 DataLoader
        val_loader: 검증 DataLoader
        device: 학습 디바이스
        lr: 초기 학습률
        weight_decay: L2 정규화 계수
        num_epochs: 최대 에폭 수
        checkpoint_dir: 체크포인트 저장 디렉토리
        log_dir: TensorBoard 로그 디렉토리
        use_amp: 혼합 정밀도 학습 사용 여부
        early_stop_patience: Early stopping 인내 에폭
    """

    def __init__(
        self,
        model: FallDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        checkpoint_dir: str = "model/weights",
        log_dir: str = "logs/tensorboard",
        use_amp: bool = True,
        early_stop_patience: int = 15,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 옵티마이저 & 스케줄러
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        # AMP
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # TensorBoard
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stop_patience, mode="max")

        self._best_f1 = 0.0
        self._epoch = 0

        logger.info(
            "Trainer 초기화: device=%s, lr=%.0e, epochs=%d, AMP=%s",
            device, lr, num_epochs, self.use_amp
        )

    def train_epoch(self) -> Dict[str, float]:
        """단일 에폭 학습."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    loss, loss_dict = self.model.compute_loss(x, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = self.model.compute_loss(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                logits, _ = self.model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증 세트 평가."""
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_labels = []

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            loss, _ = self.model.compute_loss(x, y)
            total_loss += loss.item()

            logits, _ = self.model(x)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    @staticmethod
    def _compute_metrics(labels, preds) -> Dict[str, float]:
        """Precision, Recall, F1 계산."""
        labels = [int(l) for l in labels]
        preds = [int(p) for p in preds]

        tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
        fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
        tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if labels else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        """체크포인트 저장."""
        ckpt = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": metrics,
            "fall_threshold": self.model.fall_threshold,
        }

        last_path = self.checkpoint_dir / "last_checkpoint.pth"
        torch.save(ckpt, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(ckpt, best_path)
            logger.info("최고 체크포인트 저장: F1=%.4f", metrics["f1"])

    def fit(self) -> Dict[str, float]:
        """
        전체 학습 루프 실행.

        Returns:
            최고 성능 메트릭
        """
        best_metrics = {}
        logger.info("학습 시작: %d 에폭", self.num_epochs)

        for epoch in range(1, self.num_epochs + 1):
            self._epoch = epoch
            epoch_start = time.time()

            # 학습
            train_metrics = self.train_epoch()
            # 검증
            val_metrics = self.validate()
            # 스케줄러 갱신
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 로그 출력
            logger.info(
                "Epoch %3d/%d | "
                "Train Loss=%.4f Acc=%.3f | "
                "Val Loss=%.4f F1=%.4f Prec=%.4f Rec=%.4f | "
                "LR=%.2e | %.1fs",
                epoch, self.num_epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["f1"],
                val_metrics["precision"], val_metrics["recall"],
                current_lr, epoch_time,
            )

            # TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

            # 체크포인트
            f1 = val_metrics["f1"]
            is_best = self.early_stopping.step(f1)
            if is_best:
                self._best_f1 = f1
                best_metrics = val_metrics.copy()
            self._save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.early_stopping.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        self.writer.close()
        logger.info(
            "학습 완료. 최고 F1=%.4f (Precision=%.4f, Recall=%.4f)",
            best_metrics.get("f1", 0),
            best_metrics.get("precision", 0),
            best_metrics.get("recall", 0),
        )
        return best_metrics


def parse_args() -> argparse.Namespace:
    """학습 스크립트 인자 파싱."""
    parser = argparse.ArgumentParser(description="Fall Guardian 모델 학습")
    parser.add_argument("--data_root", type=str, default="data/", help="데이터 루트 경로")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="model/weights")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true", help="AMP 비활성화")
    parser.add_argument("--mock_data", action="store_true", help="Mock 데이터 사용")
    return parser.parse_args()


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"학습 디바이스: {device}")

    # Mock 데이터 생성 (--mock_data 옵션)
    data_root = args.data_root
    if args.mock_data:
        print("Mock 데이터 생성 중...")
        tmpdir = tempfile.mkdtemp()
        FallDataset.generate_mock_dataset(
            output_dir=tmpdir,
            n_fall=100,
            n_normal=400,
        )
        data_root = tmpdir
        print(f"Mock 데이터 위치: {data_root}")

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 모델 생성
    model = FallDetector(
        input_dim=6,
        pointnet_output_dim=1024,
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_dropout=0.3,
        fall_threshold=0.7,
    )
    print(f"모델 파라미터 수: {model.count_parameters():,}")

    # 학습
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_amp=not args.no_amp,
    )

    best_metrics = trainer.fit()
    print(f"\n학습 완료!")
    print(f"최고 F1: {best_metrics.get('f1', 0):.4f}")
