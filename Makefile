# 所有命令都在仓库根执行；后面加 up/down/migrate 也进这里。
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help sync lint fmt typecheck test check config check-deps test-int test-all \
        migrate migrate-new migrate-down migrate-check gen-contracts gen-contracts-check

# 迁移统一在 packages/db 下执行（alembic.ini 在那里）
ALEMBIC ?= $(UV) run alembic -c packages/db/alembic.ini

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

migrate:  ## 升级到最新版本
	$(ALEMBIC) upgrade head

migrate-new:  ## 生成新迁移，用法：make migrate-new name=add_documents
	@test -n "$(name)" || (echo "用法: make migrate-new name=描述" && exit 1)
	$(ALEMBIC) revision --autogenerate -m "$(name)"

migrate-down:  ## 回退一个版本
	$(ALEMBIC) downgrade -1

migrate-check:  ## 检查模型与迁移是否漂移（有未生成的变更则失败）
	@$(ALEMBIC) check || (echo "" && echo "模型与迁移不一致，请运行 make migrate-new name=xxx" && exit 1)

config:  ## 打印当前生效配置（密钥脱敏）
	$(UV) run python scripts/show_config.py

gen-contracts:  ## 从 Pydantic 契约生成 JSON Schema 与前端 TS 类型
	$(UV) run python packages/contracts/scripts/export_schema.py
	$(UV) run python packages/contracts/scripts/gen_ts.py

gen-contracts-check:  ## 检查生成产物是否与源头契约漂移
	$(UV) run python packages/contracts/scripts/check_contracts_fresh.py

check: lint typecheck gen-contracts-check test  ## 提交前全量自检（含契约漂移检查）
