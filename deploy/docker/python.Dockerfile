# Python 应用镜像：API / Worker / 迁移共用。
#
# ## 为什么一个镜像跑三个角色
#
# API 与 Worker 依赖完全相同的 workspace 包（contracts、core、db、adapters），
# 唯一区别是启动命令。拆成两个镜像会让 torch/bge 模型依赖被复制两份
# （镜像多出数 GB），且版本漂移风险更高。
#
# ## 为什么分 builder / runtime 两段
#
# `uv sync` 需要下载与解包大量 wheel（torch 尤其大）；最终镜像只保留
# 安装好的 venv 与源码。编译工具链（build-essential）不会进运行镜像。
#
# ## torch 后端
#
# 镜像严格按 `uv.lock` 安装（`--frozen`），因此 torch wheel 以锁文件为准
# （当前锁是 PyPI 默认的 CUDA 版）。CPU-only 机器上 torch 仍会自动回退到
# CPU，只是镜像更大、下载更久；不再额外做 CPU/CUDA 双 lock。
#
# GPU 机器只需要声明设备（compose.gpu.yml），镜像本身已带 CUDA 运行时。
#
# ## 非 root
#
# 运行用户 uid/gid=10001。命名卷（HuggingFace 缓存）首次挂载时会继承
# 镜像内目录的所有者，因此不需要 entrypoint 里再 chown。

# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

# 为什么不用 ghcr.io/astral-sh/uv 镜像：
# 部分网络环境无法直连 ghcr.io，而 Docker Hub 通常已有镜像加速。
# 用 pip 固定版本安装 uv，依赖来源与项目其它依赖一致。
ARG UV_VERSION=0.13.8
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# 需要 C 工具链的场景：个别依赖只有 sdist（如 jieba 的 setup.py）。
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 先只拷 manifest 再 sync：依赖不变时这一层可命中缓存。
# 注意不能只拷根 pyproject：workspace 成员的 pyproject 也参与解析。
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY apps/api ./apps/api
COPY apps/worker ./apps/worker

# --no-dev：不装 pytest/ruff/mypy 等开发依赖。
# alembic 是 rag-db 的运行时依赖，迁移服务因此可用。
RUN uv sync --all-packages --no-dev --frozen

# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

# curl 仅用于容器 healthcheck。
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home rag

WORKDIR /app

# venv 内的 workspace 包是 editable 安装，.pth 指向 /app/packages 与
# /app/apps；因此这两个目录必须与 builder 中路径一致。
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/packages /app/packages
COPY --from=builder /app/apps /app/apps
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/

# 运维脚本（回填 search_text 等）随镜像提供，便于：
#   docker compose exec api python /app/scripts/backfill_search_text.py
COPY scripts ./scripts

# HuggingFace 缓存目录：命名卷挂载点，必须先存在且属于运行用户。
# 本地存储目录必须属于运行用户：
# 命名卷首次挂载时会继承镜像内目录的所有者；不存在的话 Docker 会创建
# 一个 root 拥有的空目录，worker 启动时 mkdir 直接 Permission denied。
RUN mkdir -p /home/rag/.cache/huggingface /app/.data/storage \
    && chown -R rag:rag /home/rag /app

USER rag

EXPOSE 8000

# 默认启动 API；Worker 与迁移在 compose 里用 command 覆盖。
CMD ["uvicorn", "--factory", "rag_api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
