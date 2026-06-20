#!/usr/bin/env bash
# ============================================================
# setup_jetson_nano.sh
# ============================================================
# 구형 NVIDIA Jetson Nano (4GB, JetPack 4.6.x) 에서 Fall Guardian을
# 실행하기 위한 시스템 패키지 / Python 3.8 / venv 준비 스크립트.
#
# 전제조건:
#   - JetPack 4.6.x (L4T R32.7.x) 가 SD카드에 플래시되어 있어야 함
#   - 인터넷 연결
#   - 빌드 중 메모리 부족을 막기 위해 8GB 이상의 스왑 공간 권장
#     (Jetson Nano 4GB RAM은 PyTorch/Python 소스 빌드에 부족함)
#
# 사용:
#   chmod +x scripts/setup_jetson_nano.sh
#   ./scripts/setup_jetson_nano.sh
#
# 이 스크립트는 다음을 수행한다:
#   1. nvpmodel 고성능 모드 + jetson_clocks 적용
#   2. 8GB 스왑 파일 생성 (없는 경우)
#   3. 빌드/런타임에 필요한 apt 패키지 설치
#   4. Python 3.8을 소스에서 빌드 (JetPack 4.6 기본은 3.6.9)
#   5. --system-site-packages venv 생성 (TensorRT 시스템 패키지 참조용)
#   6. requirements-jetson-nano.txt 설치
#   7. pycuda 소스 빌드
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.8.18"
VENV_DIR="${PROJECT_ROOT}/.venv"

log() { echo -e "\n\033[1;32m[setup_jetson_nano]\033[0m $*"; }

if ! grep -qi "tegra" /proc/version 2>/dev/null && [[ "$(uname -m)" != "aarch64" ]]; then
    log "경고: aarch64/Tegra 환경이 아닙니다. Jetson Nano가 아닌 곳에서 실행 중인지 확인하세요."
fi

# ── 1. 성능 모드 ────────────────────────────────
log "1/7 nvpmodel 고성능 모드 + jetson_clocks 설정"
if command -v nvpmodel >/dev/null 2>&1; then
    sudo nvpmodel -m 0 || true
    sudo jetson_clocks || true
else
    log "  nvpmodel 없음 - 건너뜀 (Jetson 보드가 아닐 수 있음)"
fi

# ── 2. 스왑 ──────────────────────────────────────
log "2/7 스왑 공간 확인 (8GB 목표)"
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
if [[ "$SWAP_TOTAL_KB" -lt 7000000 ]]; then
    log "  스왑 부족 (${SWAP_TOTAL_KB}KB) -> /swapfile 8GB 생성"
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q "/swapfile" /etc/fstab; then
        echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
    fi
else
    log "  스왑 충분 (${SWAP_TOTAL_KB}KB) - 건너뜀"
fi

# ── 3. apt 패키지 ────────────────────────────────
log "3/7 apt 패키지 설치 (빌드 도구, 시리얼, 오디오, ffmpeg 등)"
sudo apt-get update
sudo apt-get install -y \
    build-essential checkinstall \
    libreadline-gdbm-dev libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    libffi-dev zlib1g-dev liblzma-dev \
    libopenblas-dev libopenmpi-dev \
    espeak-ng ffmpeg portaudio19-dev \
    python3-tensorrt \
    cmake git wget

# ── 4. Python 3.8 소스 빌드 ──────────────────────
log "4/7 Python ${PYTHON_VERSION} 빌드 확인"
if command -v python3.8 >/dev/null 2>&1; then
    log "  python3.8 이미 설치되어 있음 - 건너뜀"
else
    BUILD_DIR="/tmp/python-build"
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    wget -nc "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    tar -xf "Python-${PYTHON_VERSION}.tgz"
    cd "Python-${PYTHON_VERSION}"
    ./configure --enable-optimizations --with-ensurepip=install
    # Jetson Nano 4코어 -> 빌드 시간 약 1~2시간 소요
    make -j"$(nproc)"
    sudo make altinstall   # /usr/bin/python3 를 덮어쓰지 않음
    cd "$PROJECT_ROOT"
fi

# ── 5. venv 생성 (시스템 site-packages 참조: tensorrt 바인딩 사용) ──
log "5/7 venv 생성: ${VENV_DIR} (--system-site-packages, TensorRT 바인딩 참조용)"
if [[ ! -d "$VENV_DIR" ]]; then
    python3.8 -m venv --system-site-packages "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

# ── 6. Python 의존성 ─────────────────────────────
log "6/7 requirements-jetson-nano.txt 설치"
pip install -r "${PROJECT_ROOT}/requirements-jetson-nano.txt"

log "  PyTorch/torchvision은 NVIDIA Jetson 전용 wheel을 별도로 설치해야 합니다:"
log "  https://forums.developer.nvidia.com/t/pytorch-for-jetson 에서"
log "  JetPack 4.6.x + Python 3.8 조합의 wheel(.whl)을 다운로드한 뒤"
log "  '${VENV_DIR}/bin/pip install <다운로드한 torch wheel>' 로 설치하세요."

# ── 7. pycuda 소스 빌드 ──────────────────────────
log "7/7 pycuda 빌드 (CUDA_ROOT=/usr/local/cuda)"
export CUDA_ROOT=/usr/local/cuda
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
pip install pycuda

log "설치 완료. 다음 단계:"
log "  1) source ${VENV_DIR}/bin/activate"
log "  2) PyTorch wheel 설치 (위 안내 참고)"
log "  3) config/iwr6843_profile.cfg 를 실제 mmWave Demo Visualizer export 파일로 교체"
log "  4) python main.py --mock --debug 로 동작 확인"
log "  5) python main.py --config config/config.yaml 로 실하드웨어 실행"
