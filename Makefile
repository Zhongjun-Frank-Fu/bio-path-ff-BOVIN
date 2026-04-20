# ============================================================================
# BOVIN-Pathway Demo · Makefile
# 目标：让 README 的 "Quickstart" 就是这个 Makefile 的 3 个 target
# ============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE := bovin-pathway-demo
TAG   := 0.1.0
PLATFORM ?= linux/amd64
SEED ?= 42
CONFIG ?= configs/tcga_coad.yaml

.PHONY: help install lint fmt test smoke docker-build docker-run docker-shell clean

help:                 ## 显示所有 target
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ============== 本地开发 ==============
install:              ## 本地 pip install -e .[dev]
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"
	pre-commit install

lint:                 ## Ruff 检查
	ruff check bovin_demo tests

fmt:                  ## Ruff format
	ruff format bovin_demo tests
	ruff check --fix bovin_demo tests

test:                 ## 跑 pytest
	pytest -v

smoke:                ## import smoke
	python -c "import bovin_demo; print(bovin_demo.__version__)"

data-coad:            ## M2 · T2.1 — 下载 TCGA-COAD RNA-seq + clinical + survival
	bash tools/download_tcga_coad.sh

data-coad-force:      ## 强制重新下载（覆盖 data/raw/*.gz）
	bash tools/download_tcga_coad.sh --force

# ============== Docker ==============
docker-build:         ## docker build
	docker build --platform=$(PLATFORM) -t $(IMAGE):$(TAG) .

docker-run:           ## 容器里 import smoke
	docker run --rm --platform=$(PLATFORM) $(IMAGE):$(TAG) \
		python -c "import bovin_demo; print('[docker] bovin_demo v'+bovin_demo.__version__+' OK')"

docker-shell:         ## 进入容器 bash
	docker run --rm -it --platform=$(PLATFORM) -v "$$PWD:/app" $(IMAGE):$(TAG) bash

# ============== 训练 / 评估 ==============
train:                ## M4 — 单 seed 训练（默认 seed=42）
	python -m bovin_demo.cli train --config $(CONFIG) --seed $(SEED)

train-sweep:          ## M4 · T4.5 — 三 seed 稳定性 sweep（42/1337/2024）
	python -m bovin_demo.cli train --config $(CONFIG) --seeds 42,1337,2024

RUN_DIR ?= $(shell ls -dt outputs/*/ 2>/dev/null | head -1 | sed 's:/$$::')

xai:                  ## M5 — IG + heatmap 对最近一次 run
	python -m bovin_demo.cli xai --run-dir $(RUN_DIR) --config $(CONFIG)

eval:                 ## M6 — 汇总 metrics + XAI → report.md
	python -m bovin_demo.cli eval --run-dir $(RUN_DIR) --config $(CONFIG)

eval-luad:            ## M6 · T6.4 — eval + LUAD 零样本迁移
	python -m bovin_demo.cli eval --run-dir $(RUN_DIR) --config $(CONFIG) --luad-raw-dir data/raw_luad

data-luad:            ## M6 · T6.4 — 下载 TCGA-LUAD
	bash tools/download_tcga_luad.sh

# ============== 清理 ==============
clean:                ## 清缓存
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info
