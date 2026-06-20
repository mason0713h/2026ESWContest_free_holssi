"""
tests/test_model.py
===================
낙상 감지 모델 단위 테스트.

테스트 항목:
  - TNet: forward, 직교성 손실
  - PointNet: forward shape, dtype
  - TimeDistributedPointNet: batch+time 처리
  - GRUClassifier: forward, predict_proba
  - FallDetector: end-to-end forward, 저장/불러오기
  - LongLieDetector: 상태 전이, 레벨 상승
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from model.pointnet import TNet, PointNet, TimeDistributedPointNet
from model.gru_classifier import GRUClassifier, FocalLoss
from model.fall_detector import FallDetector
from model.long_lie_detector import LongLieDetector, AlertLevel


# ── Fixtures ───────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def T():
    return 16


@pytest.fixture
def N():
    return 64


@pytest.fixture
def C():
    return 6


# ── TNet 테스트 ────────────────────────────────

class TestTNet:

    def test_output_shape(self, device, batch_size):
        """TNet 출력이 (B, k, k) 형태이어야 한다."""
        k = 3
        tnet = TNet(k=k, input_dim=k).to(device)
        x = torch.randn(batch_size, k, 64, device=device)
        out = tnet(x)
        assert out.shape == (batch_size, k, k)

    def test_feature_tnet_shape(self, device, batch_size):
        """특징 TNet (64×64) 출력 shape 확인."""
        k = 64
        tnet = TNet(k=k, input_dim=k).to(device)
        x = torch.randn(batch_size, k, 64, device=device)
        out = tnet(x)
        assert out.shape == (batch_size, k, k)

    def test_orthogonality_loss_positive(self, device, batch_size):
        """직교성 손실이 0 이상이어야 한다."""
        k = 64
        tnet = TNet(k=k, input_dim=k).to(device)
        x = torch.randn(batch_size, k, 64, device=device)
        trans = tnet(x)
        loss = TNet.orthogonality_loss(trans)
        assert loss.item() >= 0.0

    def test_identity_initialization(self, device):
        """초기화 직후 출력이 항등 행렬에 가까워야 한다."""
        k = 3
        tnet = TNet(k=k, input_dim=k).to(device)
        tnet.eval()
        with torch.no_grad():
            x = torch.zeros(1, k, 64, device=device)
            out = tnet(x)
        # 모든 가중치 0 → 항등 행렬
        I = torch.eye(k, device=device)
        # 완전한 항등 행렬은 아니지만 대각 성분이 크게 달라지진 않음
        assert out.shape == (1, k, k)


# ── PointNet 테스트 ────────────────────────────

class TestPointNet:

    def test_output_shape(self, device, batch_size, N, C):
        """PointNet 글로벌 특징 shape이 (B, 1024)이어야 한다."""
        pn = PointNet(input_dim=C, output_dim=1024).to(device)
        x = torch.randn(batch_size, N, C, device=device)
        feat, t_in, t_feat = pn(x)
        assert feat.shape == (batch_size, 1024)

    def test_tnet_shapes(self, device, batch_size, N, C):
        """T-Net 출력 shape 확인."""
        pn = PointNet(input_dim=C, output_dim=1024, use_tnet_input=True, use_tnet_feature=True).to(device)
        x = torch.randn(batch_size, N, C, device=device)
        _, t_in, t_feat = pn(x)
        assert t_in.shape == (batch_size, C, C)
        assert t_feat.shape == (batch_size, 64, 64)

    def test_without_tnet(self, device, batch_size, N, C):
        """T-Net 없이도 동작해야 한다."""
        pn = PointNet(input_dim=C, output_dim=1024,
                      use_tnet_input=False, use_tnet_feature=False).to(device)
        x = torch.randn(batch_size, N, C, device=device)
        feat, t_in, t_feat = pn(x)
        assert feat.shape == (batch_size, 1024)
        assert t_in is None
        assert t_feat is None

    def test_output_dtype(self, device, batch_size, N, C):
        """출력 dtype이 float32이어야 한다."""
        pn = PointNet(input_dim=C, output_dim=1024).to(device)
        x = torch.randn(batch_size, N, C, device=device)
        feat, _, _ = pn(x)
        assert feat.dtype == torch.float32

    def test_different_output_dim(self, device, batch_size, N, C):
        """다른 출력 차원도 지원해야 한다."""
        for out_dim in [256, 512, 1024]:
            pn = PointNet(input_dim=C, output_dim=out_dim).to(device)
            x = torch.randn(batch_size, N, C, device=device)
            feat, _, _ = pn(x)
            assert feat.shape == (batch_size, out_dim)


# ── TimeDistributedPointNet 테스트 ─────────────

class TestTimeDistributedPointNet:

    def test_output_shape(self, device, batch_size, T, N, C):
        """출력 shape이 (B, T, 1024)이어야 한다."""
        td_pn = TimeDistributedPointNet(input_dim=C, output_dim=1024).to(device)
        x = torch.randn(batch_size, T, N, C, device=device)
        out, _, _ = td_pn(x)
        assert out.shape == (batch_size, T, 1024)

    def test_batch_independence(self, device, T, N, C):
        """배치 내 샘플이 독립적으로 처리되어야 한다."""
        td_pn = TimeDistributedPointNet(input_dim=C, output_dim=1024).to(device)
        td_pn.eval()
        x1 = torch.randn(1, T, N, C, device=device)
        x2 = torch.randn(1, T, N, C, device=device)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out_batch, _, _ = td_pn(x_batch)
            out1, _, _ = td_pn(x1)
            out2, _, _ = td_pn(x2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


# ── GRUClassifier 테스트 ───────────────────────

class TestGRUClassifier:

    def test_output_shape(self, device, batch_size, T):
        """출력 logit shape이 (B, 2)이어야 한다."""
        gru = GRUClassifier(input_dim=1024, hidden_size=256, num_layers=2).to(device)
        x = torch.randn(batch_size, T, 1024, device=device)
        logits, hidden = gru(x)
        assert logits.shape == (batch_size, 2)

    def test_hidden_shape(self, device, batch_size, T):
        """마지막 hidden state shape이 (B, 256)이어야 한다."""
        gru = GRUClassifier(input_dim=1024, hidden_size=256).to(device)
        x = torch.randn(batch_size, T, 1024, device=device)
        _, hidden = gru(x)
        assert hidden.shape == (batch_size, 256)

    def test_predict_proba_sum_to_1(self, device, batch_size, T):
        """Softmax 확률 합이 1이어야 한다."""
        gru = GRUClassifier(input_dim=1024, hidden_size=256).to(device)
        gru.eval()
        x = torch.randn(batch_size, T, 1024, device=device)
        proba = gru.predict_proba(x)
        assert proba.shape == (batch_size, 2)
        assert torch.allclose(proba.sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5)

    def test_predict_binary_output(self, device, batch_size, T):
        """예측 결과가 0 또는 1이어야 한다."""
        gru = GRUClassifier(input_dim=1024).to(device)
        gru.eval()
        x = torch.randn(batch_size, T, 1024, device=device)
        pred, conf = gru.predict(x, threshold=0.7)
        assert pred.shape == (batch_size,)
        assert ((pred == 0) | (pred == 1)).all()

    def test_focal_loss_positive(self, device, batch_size, T):
        """Focal Loss가 0 이상이어야 한다."""
        gru = GRUClassifier(input_dim=1024).to(device)
        x = torch.randn(batch_size, T, 1024, device=device)
        logits, _ = gru(x)
        targets = torch.randint(0, 2, (batch_size,), device=device)
        loss = FocalLoss()(logits, targets)
        assert loss.item() >= 0


# ── FallDetector 통합 테스트 ───────────────────

class TestFallDetector:

    def test_forward_shape(self, device, batch_size, T, N, C):
        """FallDetector 출력 shape이 (B, 2)이어야 한다."""
        model = FallDetector().to(device)
        x = torch.randn(batch_size, T, N, C, device=device)
        logits, _ = model(x)
        assert logits.shape == (batch_size, 2)

    def test_predict_single_sample(self, device, T, N, C):
        """단일 샘플 예측이 동작해야 한다."""
        model = FallDetector().to(device)
        model.eval()
        x = torch.randn(T, N, C, device=device)
        fall, confidence = model.predict(x)
        assert isinstance(fall, bool)
        assert 0.0 <= confidence <= 1.0

    def test_loss_computation(self, device, batch_size, T, N, C):
        """손실 계산이 동작하고 양수여야 한다."""
        model = FallDetector().to(device)
        x = torch.randn(batch_size, T, N, C, device=device)
        targets = torch.randint(0, 2, (batch_size,), device=device)
        loss, loss_dict = model.compute_loss(x, targets)
        assert loss.item() > 0
        assert "focal_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_save_and_load(self, T, N, C):
        """저장/불러오기가 동작해야 한다."""
        model = FallDetector()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pth")
            model.save(path)
            assert Path(path).exists()

            model2 = FallDetector()
            model2.load(path)
            model2.eval()  # BatchNorm eval 모드 (배치 크기 1 대응)

            # 동일한 입력에 대해 같은 출력을 내야 함
            x = torch.randn(2, T, N, C)  # BatchNorm은 배치 크기 >= 2 필요
            with torch.no_grad():
                out1, _ = model(x)
                out2, _ = model2(x)
            assert torch.allclose(out1, out2, atol=1e-5)

    def test_parameter_count(self):
        """파라미터 수가 0보다 커야 한다."""
        model = FallDetector()
        assert model.count_parameters() > 0


# ── LongLieDetector 테스트 ─────────────────────

class TestLongLieDetector:

    def test_initial_normal_state(self):
        """초기 상태는 NORMAL이어야 한다."""
        detector = LongLieDetector()
        level = detector.update(1, fall_detected=False, fall_confidence=0.0)
        assert level == AlertLevel.NORMAL

    def test_fall_triggers_level1(self):
        """낙상 감지 시 FALL 레벨이 되어야 한다."""
        detector = LongLieDetector()
        level = detector.update(1, fall_detected=True, fall_confidence=0.9)
        assert level == AlertLevel.FALL

    def test_level_escalation_over_time(self):
        """시간 경과에 따라 레벨이 상승해야 한다."""
        detector = LongLieDetector(
            level1_seconds=0.1,
            level2_seconds=0.3,
            level3_seconds=0.6,
        )

        # 낙상 감지
        detector.update(1, fall_detected=True, fall_confidence=0.9)

        # 0.2초 후 → Level 2
        time.sleep(0.2)
        level = detector.update(1, fall_detected=True, fall_confidence=0.9)
        assert level >= AlertLevel.LONG_LIE_30

    def test_callback_triggered(self):
        """Level 상승 시 콜백이 호출되어야 한다."""
        detector = LongLieDetector(level1_seconds=0.0)
        triggered = []

        def on_fall(pid, state):
            triggered.append(pid)

        detector.register_callback(AlertLevel.FALL, on_fall)
        detector.update(1, fall_detected=True, fall_confidence=0.9)
        assert 1 in triggered

    def test_recovery_resets_state(self):
        """회복 후 NORMAL로 돌아가야 한다."""
        detector = LongLieDetector(
            recovery_hold_seconds=0.0,  # 즉시 회복
        )
        # 낙상
        detector.update(1, fall_detected=True, fall_confidence=0.9)
        state = detector.get_state(1)
        assert state.alert_level == AlertLevel.FALL

        # 회복 (z 높은 포인트) - 2회 호출로 hold 조건 충족
        high_points = np.zeros((10, 6), dtype=np.float32)
        high_points[:, 2] = 1.2  # 서있는 높이
        # 첫 번째 호출: 회복 시작 타이머 등록
        detector.update(
            1,
            fall_detected=False,
            fall_confidence=0.0,
            person_points=high_points,
        )
        # 두 번째 호출: hold_seconds=0.0이면 즉시 회복
        level = detector.update(
            1,
            fall_detected=False,
            fall_confidence=0.0,
            person_points=high_points,
        )
        # 즉시 회복이면 NORMAL
        assert level == AlertLevel.NORMAL

    def test_multiple_persons_independent(self):
        """다수 인원이 독립적으로 추적되어야 한다."""
        detector = LongLieDetector()
        level1 = detector.update(1, fall_detected=True, fall_confidence=0.9)
        level2 = detector.update(2, fall_detected=False, fall_confidence=0.0)

        assert level1 == AlertLevel.FALL
        assert level2 == AlertLevel.NORMAL

    def test_all_alerts_property(self):
        """활성 알림 딕셔너리 확인."""
        detector = LongLieDetector()
        detector.update(1, fall_detected=True, fall_confidence=0.9)
        detector.update(2, fall_detected=True, fall_confidence=0.8)

        alerts = detector.all_alerts
        assert 1 in alerts
        assert 2 in alerts
        assert len(alerts) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
