# 서류 제출용 그래프

`scripts/generate_report_figures.py` 실행 결과 (`python scripts/generate_report_figures.py`).
모두 **mock 데이터 + 학습되지 않은 랜덤 가중치** 기준으로 측정 가능한 항목만 다루며,
실제 라벨링 데이터나 실보드(Jetson Nano) 성능 수치(F1, TensorRT 가속 효과 등)는
포함하지 않는다 — 허위 수치를 보고하지 않기 위한 의도적인 제한이다.

## fig1_pipeline_latency.png
파이프라인 3단계(전처리 DBSCAN+정규화 / 칼만 필터 추적 / 모델 추론)의 처리 시간을
60개 윈도우에 대해 측정한 box plot. **개발 PC(샌드박스) CPU, PyTorch eager 모드** 기준이며
Jetson Nano TensorRT 추론 시간이 아님을 제목에 명시.

## fig2_dbscan_noise_ratio.png
"서있음(normal)" vs "낙상 후 바닥(fall)" 시나리오별로 DBSCAN이 전체 포인트 중
노이즈로 제거한 비율의 분포를 box plot으로 비교. 두 시나리오에서 제거율 분포가
다르게 나타나는지(레이더 반사 패턴 차이) 확인하는 용도.

## fig3_longlie_timeline.png
낙상 후 자력 회복 없이 150초간 바닥에 누워있는 가상 시나리오를 시간을 가속해
재생, `LongLieDetector`가 30초/60초/120초 경과 시점에 알림 레벨을
L1→L2→L3→L4(응급)로 단계적으로 올리는 과정을 계단형 그래프로 시각화.

## fig4_model_params.png
`FallDetector`(총 4.24M 파라미터)를 모듈별(T-Net 입력/특징, PointNet conv/BN, GRU,
분류 MLP)로 분해해 막대그래프로 표현. 모델 어디에 파라미터가 집중되는지 보여준다.

## fig5_onnx_size.png
동일한(학습되지 않은) 가중치를 PyTorch `state_dict(.pt)`와 ONNX(opset 13, `.onnx`)로
각각 저장했을 때 파일 크기를 비교. TensorRT FP16/INT8 양자화 **이전** 단계의 크기이며,
실제 배포 시 TRT 엔진 크기는 이보다 작아질 수 있음.

## fig6_denoising_scatter.png
Mock 낙상 프레임 1개에 DBSCAN 디노이징을 적용해, 유지된 포인트(파란 점)와
노이즈로 제거된 포인트(빨간 ×)를 3D 산점도로 비교. eps/min_samples 파라미터가
실제로 어떤 포인트를 걸러내는지 직관적으로 보여준다.

## fig7_multi_tracker_trajectory.png
가상의 2인이 각각 직선/원형 경로로 동시에 이동하는 상황을 합성 포인트클라우드로
만들어 `MultiPersonTracker`에 50프레임 동안 입력, 칼만 필터가 두 사람의 트랙 ID를
섞지 않고 끝까지 유지하며 추적하는지 궤적으로 시각화 (▲ 시작점, ■ 끝점).

## fig8_model_architecture.png
`FallDetector` 코드에서 실제 레이어 차원(PointNet `output_dim=1024`, GRU
`hidden_size=256, layers=2` 등)을 직접 읽어와 그린 데이터 흐름 블록 다이어그램.
입력 `(B,16,64,6)` → PointNet → GRU → 분류 MLP → Softmax `(B,2)`까지의 전체 흐름.
