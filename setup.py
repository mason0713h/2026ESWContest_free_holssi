"""
setup.py
========
Fall Guardian 패키지 설정.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="fall_guardian",
    version="1.0.0",
    description="4D mmWave 레이더 기반 낙상 감지 시스템 (임베디드SW경진대회)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fall Guardian Team",
    # Jetson Orin Nano(JetPack 6.x)는 Python 3.10+, 구형 Jetson Nano(JetPack
    # 4.6.x)는 소스 빌드한 Python 3.8을 사용한다 (scripts/setup_jetson_nano.sh).
    python_requires=">=3.8",
    packages=find_packages(
        exclude=["tests*", "scripts*", "data*", "logs*"]
    ),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyserial>=3.5",
        "PyYAML>=6.0",
        "paho-mqtt>=1.6.1",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "full": [
            "twilio>=8.0.0",
            "openai-whisper>=20231117",
            "pyttsx3>=2.90",
            "sounddevice>=0.4.6",
            "onnx>=1.15.0",
            "onnxruntime>=1.16.0",
            "tensorboard>=2.14.0",
            "matplotlib>=3.7.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fall-guardian=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
