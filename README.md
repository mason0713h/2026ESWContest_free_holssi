# 낙상 가디언 (Fall Guardian)

**제24회 임베디드 소프트웨어 경진대회 자유공모 부문 출품작**  
Team: 홀씨 | 팀장: 하승호 (경기대학교 AI컴퓨터공학부 1학년)  
마감: 2026년 9월 3일

---

## 프로젝트 개요

가정용 **천장형 온디바이스 레이더 낙상감지·자동알림 시스템**입니다.

4D mmWave 레이더(TI IWR6843ISK-ODS)에서 포인트클라우드를 수집하고,  
실제 배포 보드인 **구형 Jetson Nano(4GB, JetPack 4.6.x)** 위에서 **TensorRT FP16/INT8**
추론으로 낙상을 실시간 감지합니다 (Jetson Orin Nano Super로 업그레이드 시
[설치 방법 B](#b-jetson-orin-nano-super-jetpack-6x-향후-업그레이드-대상) 참고).  
낙상 감지 시 음성 확인 → 보호자 앱(MQTT) → SMS → 119 웹훅 순으로 자동 에스컬레이션합니다.

### 핵심 차별화
| 기능 | FallPoint (ISEF 2025) | 상용품 | **낙상 가디언** |
|------|----------------------|--------|----------------|
| 온디바이스 추론 | ❌ 클라우드 | 부분 | ✅ TensorRT <20ms |
| Long-lie 감지 | ❌ | ❌ | ✅ 60초→SMS→119 |
| 다중 인원 추적 | ❌ 단일인 | 제한 | ✅ 칼만 필터 2인+ |
| 욕실 멀티모달 | 제한 | Vayyar만 | ✅ mmWave + 진동 |
| 오픈소스 | ❌ | ❌ | ✅ MIT License |
| 한국형 119 연동 | ❌ | ❌ | ✅ 개념실증 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                TI IWR6843ISK-ODS (60GHz)                    │
│     4D 포인트클라우드 (x, y, z, velocity) → UART → USB      │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│   Jetson Nano (4GB, JetPack 4.6.x, TRT 8.2.1) — 실제 배포 타깃  │
│   (Jetson Orin Nano Super로 업그레이드 시 설치 방법 B 참고)      │
│                                                             │
│  [radar/capture.py]   → 슬라이딩 윈도우 16프레임 버퍼       │
│  [radar/preprocessor.py] → Dynamic DBSCAN + 64점 정규화    │
│  [radar/multi_tracker.py] → 칼만 필터 다중 인원 추적        │
│                                                             │
│  [model/pointnet.py]  → Time-distributed 공간 특징 추출     │
│  [model/gru_classifier.py] → GRU 시간 패턴 분류            │
│  [model/tensorrt_inference.py] → TRT INT8 <20ms 추론        │
│  [model/long_lie_detector.py] → Long-lie 30/60/120초 감지  │
│                                                             │
│  [alert/voice_confirm.py] → TTS "괜찮으세요?" + STT 응답    │
│  [alert/mqtt_publisher.py] → 보호자 앱 실시간 알림          │
│  [alert/escalation.py] → SMS(Twilio) → 119 웹훅            │
└─────────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조

```
fall_guardian/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml            # 레이더 파라미터, MQTT 설정
├── radar/
│   ├── capture.py             # TI IWR6843 UART 통신, TLV 파싱
│   ├── preprocessor.py        # Dynamic DBSCAN, 오버샘플링, 정규화
│   └── multi_tracker.py       # 칼만 필터 다중 인원 추적
├── model/
│   ├── pointnet.py            # Time-distributed PointNet (T-Net 포함)
│   ├── gru_classifier.py      # GRU + MLP + FocalLoss
│   ├── fall_detector.py       # 통합 모델 (4.2M 파라미터)
│   ├── long_lie_detector.py   # Long-lie 단계별 알림 레벨
│   └── tensorrt_inference.py  # TensorRT FP16/INT8 추론 엔진
├── training/
│   ├── dataset.py             # FallDataset (회전/지터 증강)
│   ├── train.py               # AdamW + CosineAnnealing 학습
│   ├── evaluate.py            # F1/Precision/Recall/ROC-AUC
│   └── export_onnx.py         # ONNX 변환 스크립트
├── alert/
│   ├── mqtt_publisher.py      # MQTT 낙상 알림 발행
│   ├── voice_confirm.py       # 한국어 TTS + Whisper STT
│   └── escalation.py          # L1 MQTT → L2 SMS → L3 119
├── scripts/
│   ├── collect_data.py        # 데이터 수집 스크립트
│   ├── benchmark_latency.py   # TensorRT 레이턴시 벤치마크
│   └── visualize_pointcloud.py # 포인트클라우드 시각화
├── tests/
│   ├── test_capture.py         # TLV 파싱, FrameWindow 슬라이딩 윈도우
│   ├── test_preprocessor.py
│   ├── test_model.py
│   ├── test_multi_tracker.py
│   ├── test_tensorrt_inference.py  # TRT 폴백 체인(ONNX Runtime/랜덤)
│   └── test_alert.py
└── main.py                    # asyncio 전체 파이프라인
```

---

## 하드웨어 요구사항

| 부품 | 모델 | 역할 | 가격 |
|------|------|------|------|
| 레이더 | TI IWR6843ISK-ODS | 4D 포인트클라우드 | ~$100 |
| 엣지 보드 | NVIDIA Jetson Nano (4GB, JetPack 4.6.x) — 실제 배포 타깃 | TensorRT 추론 | ~$99 |
| 진동 센서 | MPU-6050 (욕실용) | 낙상 충격 트리거 | ~$2 |
| 케이블 | USB-UART | 레이더↔Jetson 연결 | ~$5 |

> 향후 Jetson Orin Nano Super (8GB)로 업그레이드 시 설치 방법 B 참고.

---

## 설치 방법

### A. 구형 Jetson Nano (4GB, JetPack 4.6.x) — 실제 배포 타깃

JetPack 4.6.x는 Python 3.6.9 / TensorRT 8.2.1 / CUDA 10.2를 탑재하고 있어
최신 numpy/scipy/openai-whisper 등을 그대로 설치할 수 없다. 아래 스크립트가
Python 3.8 빌드, venv 생성, TensorRT 시스템 패키지 연결, pycuda 빌드까지
한 번에 처리한다.

```bash
chmod +x scripts/setup_jetson_nano.sh
./scripts/setup_jetson_nano.sh
source .venv/bin/activate

# PyTorch는 PyPI에 Jetson용 wheel이 없으므로 NVIDIA Jetson 전용 wheel을 받아 설치
# https://forums.developer.nvidia.com/t/pytorch-for-jetson 에서 JetPack 4.6.x +
# Python 3.8 조합 wheel(.whl)을 다운로드한 뒤:
pip install <다운로드한 torch-*.whl>
pip install <다운로드한 torchvision-*.whl>
```

레이더 설정은 반드시 실제 mmWave Demo Visualizer로 export한 `.cfg`를
`config/iwr6843_profile.cfg`에 덮어쓰고, `config/config.yaml`의
`radar.cfg_file` 경로를 확인할 것 (기본 내장값은 검증되지 않은 예시값).

USB 시리얼 포트 순서가 재부팅마다 바뀌는 문제를 막으려면
`deploy/99-iwr6843-radar.rules`를 참고해 udev 규칙으로 고정할 것.

부팅 시 자동 실행하려면:
```bash
sudo cp deploy/fall-guardian.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fall-guardian.service
```

### B. Jetson Orin Nano Super (JetPack 6.x, 향후 업그레이드 대상)
```bash
pip install -r requirements.txt
pip install -e .
# JetPack 6.x에 TensorRT 10.x 포함됨 — pycuda만 추가 설치
pip install pycuda
```

### 공통: 설정 파일 수정
```bash
cp config/config.yaml config/config_local.yaml
# config_local.yaml에서 시리얼 포트, .cfg 경로, MQTT 브로커 주소 수정
```

---

## 실행 방법

### 실제 하드웨어 모드
```bash
# 레이더 연결 후
python main.py --config config/config_local.yaml
```

### Mock 모드 (하드웨어 없이 테스트)
```bash
python main.py --mock
python main.py --mock --debug   # 디버그 로그 포함
```

### 모델 학습
```bash
# 1. 데이터 수집 (raw 레이더 .npy는 data/ 아래에 저장되며, 개인정보 보호를
#    위해 .gitignore로 전체 제외되어 있음)
python scripts/collect_data.py --output data/

# 2. 학습 (mock 데이터로 파이프라인만 검증하려면 --mock_data 추가)
python -m training.train --data_root data/ --epochs 100

# 3. ONNX 변환 (opset 13 고정 — Jetson Nano TRT 8.2.1 onnx-tensorrt 파서 제약)
python -m training.export_onnx --checkpoint model/weights/best_checkpoint.pth \
    --output model/weights/fall_guardian.onnx --opset 13

# 4. TensorRT 엔진 빌드 (model/tensorrt_inference.py의 TRTEngineBuilder 사용,
#    또는 Jetson Nano 위에서 trtexec)
python -m training.export_onnx --checkpoint model/weights/best_checkpoint.pth --export_trt
```

### 레이턴시 벤치마크
```bash
# model/weights/fall_guardian.trt 엔진을 고정 경로로 로드
python scripts/benchmark_latency.py --n_iter 100 --warmup 20
```

---

## 모델 성능 목표

| 지표 | 목표 | 참고 (FallPoint) |
|------|------|-----------------|
| F1 Score | ≥ 0.931 | 0.915 |
| Precision | ≥ 0.931 | 0.931 |
| Recall | ≥ 0.900 | 0.900 |
| 추론 레이턴시 | ≤ 20ms | N/A (클라우드) |
| 알림 응답시간 | ≤ 3초 | N/A |

---

## 테스트 실행

```bash
pytest tests/ -v
# 134개 단위 테스트
```

---

## 기술 스택

- **언어**: 개발 PC는 Python 3.11(학습/온라인 테스트), 실제 배포 타깃인
  Jetson Nano는 JetPack 4.6.x 제약으로 Python 3.8(설치 방법 A 참고)
- **AI 프레임워크**: PyTorch 2.x(학습), ONNX(opset 13) → TensorRT 8.2.1(Jetson Nano 추론).
  Orin Nano Super 업그레이드 시 TensorRT 10.x
- **레이더**: Texas Instruments mmWave SDK
- **알림**: paho-mqtt, Twilio, pyttsx3, openai-whisper
- **추적**: scikit-learn (DBSCAN), filterpy (칼만 필터)
- **테스트**: pytest

---

## 참고 논문

1. [FallPoint — ISEF 2025 ROBO021](https://isef.net/project/robo021-fall-detection-with-4d-radar-and-deep-learning)
2. [Edge-Accelerated Fall Detection via mmWave+FPGA — IEEE BIBM 2025](https://ieeexplore.ieee.org/document/11356382/)
3. [EM-Fall: 로봇 탑재 mmWave 낙상감지 — arXiv 2606.11109](https://arxiv.org/html/2606.11109)
4. [P2MFDS: 욕실 멀티모달 — arXiv 2506.17332](https://arxiv.org/html/2506.17332v1)
5. [Post-Fall Long-lie 감지 — arXiv 2601.17710](https://arxiv.org/html/2601.17710v1)
6. [Dynamic DBSCAN 다중 인원 추적 — PMC11175272](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175272/)

---

## 라이선스

MIT License — 오픈소스 공개로 국내 의료기기 스타트업 기술 이전 기여

---

## 문의

- 대회 사무국: contest@fkii.org / 02-2046-1435
- 팀 이메일: mason0713sh@gmail.com
