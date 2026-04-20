# syntax=docker/dockerfile:1.6
# ============================================================================
# BOVIN-Pathway Demo · 可复现 Docker Image
# Stage 1: torch + torch-geometric（CPU）单独 pin
# Stage 2: 项目依赖 + 源码
#
# 目的：一条命令从干净 image 起 → `python -c "import bovin_demo"` 不报错
# Apple Silicon 用户：docker build --platform=linux/amd64 ...
# ============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    BOVIN_SEED=42

# 最小系统依赖（PyG 偶尔需要编译 scatter/sparse 的 headers；CPU wheel 下通常够）
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================
# Stage: torch (CPU) — 单独一层缓存，后续只要 torch 不改就复用
# ============================================================
# Pin setuptools < 70. pytorch-lightning 2.2 imports ``pkg_resources`` via
# lightning_fabric's namespace shim; setuptools 70+ removed it and lightning
# 2.2 never backported a fix. Without this pin a fresh build fails with
# ``ModuleNotFoundError: No module named 'pkg_resources'`` at first import.
RUN pip install --upgrade pip 'setuptools<70' wheel && \
    pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1

# ============================================================
# Stage: PyG — 依赖 torch，必须放 torch 之后
# ============================================================
RUN pip install \
      torch-geometric==2.5.3 \
      torch-scatter \
      torch-sparse \
      -f https://data.pyg.org/whl/torch-2.3.1+cpu.html || \
    pip install torch-geometric==2.5.3
# Note: torch-scatter / torch-sparse 是 PyG 的可选加速包；
# 若 wheel 不可得，PyG 2.5 本身已包含 fallback 实现，不影响 demo。

# ============================================================
# Stage: 项目依赖
# ============================================================
COPY pyproject.toml /app/
# setuptools was already pinned < 70 above — do *not* upgrade here.
RUN mkdir -p /app/bovin_demo && touch /app/bovin_demo/__init__.py
RUN pip install -e ".[dev]"

# ============================================================
# Stage: 拷入真源码（最后一层，改代码只重建这层）
# ============================================================
COPY . /app/

# ============================================================
# Health check: import bovin_demo + 打印版本
# ============================================================
RUN python -c "import bovin_demo; print(f'[healthcheck] bovin_demo v{bovin_demo.__version__} loaded')"

CMD ["python", "-m", "bovin_demo.cli", "--help"]
