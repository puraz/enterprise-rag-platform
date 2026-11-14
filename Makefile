# 所有命令都在仓库根执行；后面加 up/down/migrate 也进这里。
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help sync lint fmt typecheck test check config check-deps test-int test-all

help:  ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## 同步全部 workspace 依赖（必须 --all-packages，否则四个包都不会装）
	$(UV) sync --all-packages

lint:  ## 静态检查：规则 + 格式（只查不改）
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:  ## 自动修格式 + 可自动修的规则
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## mypy 严格模式
	$(UV) run mypy

test:  ## 单元测试（默认排除 integration，无需外部依赖）
	$(UV) run pytest

test-int:  ## 集成测试（需真实 PG/Redis，先跑 make check-deps）
	$(UV) run pytest -m integration

test-all:  ## 全量测试（单测 + 集成）
	$(UV) run pytest -m ""

check-deps:  ## 检查 PG / Redis / pgvector 连通性
	$(UV) run python scripts/check_deps.py

config:  ## 打印当前生效配置（密钥脱敏）
	$(UV) run python scripts/show_config.py

check: lint typecheck test  ## 提交前全量自检
