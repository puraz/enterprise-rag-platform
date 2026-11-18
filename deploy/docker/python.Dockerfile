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
# PyPI 镜像（可选）：国内直连 pypi.org 可能只有几百 B/s。
# 与下面的 PYPI_MIRROR 用同一个值即可。
ARG PIP_INDEX_URL=""
RUN pip install --no-cache-dir \
    ${PIP_INDEX_URL:+--index-url ${PIP_INDEX_URL}} \
    "uv==${UV_VERSION}"

# 需要 C 工具链的场景：个别依赖只有 sdist（如 jieba 的 setup.py）。
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# ---- 依赖层：只拷 manifest ----
#
# ## 为什么拆成 "先 manifest、再源码" 两层
#
# 初版是 `COPY packages ./packages` 后直接 `uv sync`：改一行 Python 代码
# 就会让这一层失效，重新执行整次同步（解包、链接几百个包）。
# 拆开后，只要 `uv.lock` 与各 pyproject 不变，依赖层就命中缓存，
# 改源码只需要重装 5 个 workspace 包（都是 editable，秒级）。
COPY pyproject.toml uv.lock ./
COPY packages/contracts/pyproject.toml packages/contracts/
COPY packages/core/pyproject.toml packages/core/
COPY packages/db/pyproject.toml packages/db/
COPY packages/adapters/pyproject.toml packages/adapters/
COPY apps/api/pyproject.toml apps/api/
COPY apps/worker/pyproject.toml apps/worker/

# 可选 PyPI 镜像：
#   docker compose build --build-arg PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
#
# 为什么改 lock 而不是设 `UV_DEFAULT_INDEX`：`--frozen` 下 uv 按 lockfile
# 里记录的 registry 下载，环境变量不生效（实测：把索引指向不可达地址，
# 同步仍然成功）。只替换下载主机，版本与 sha256 哈希不变。
ARG PYPI_MIRROR=""
ARG PYPI_MIRROR_IP=""
# 用 python 而不是 sed：这条命令文本与旧版本不同，会强制重建缓存层。
#
# **必须同时改写 `files.pythonhosted.org`**：uv.lock 不仅记录 registry，
# 还记录了每个 wheel 的绝对下载地址。只改 registry 的话，元数据走镜像、
# 大文件仍从 pythonhosted 拉——实测只有几 KB/s，表现为"构建永远卡在 Downloading"。
#
# PYPI_MIRROR_IP 可选：把镜像站的 CDN 固定到一个快节点（国内 CDN 节点间
# 速度可能差两个数量级）。为空则用 DNS 默认。
RUN echo "pypi_mirror=${PYPI_MIRROR:-<empty>} pypi_ip=${PYPI_MIRROR_IP:-<dns>}" && \
    if [ -n "$PYPI_MIRROR_IP" ] && [ -n "$PYPI_MIRROR" ]; then \
        host=$(python -c "import os,urllib.parse; print(urllib.parse.urlsplit(os.environ['PYPI_MIRROR']).hostname or '')") && \
        [ -n "$host" ] && echo "$PYPI_MIRROR_IP $host" >> /etc/hosts; \
    fi && \
    if [ -n "$PYPI_MIRROR" ]; then \
        python -c "import os,pathlib; \
        m=os.environ['PYPI_MIRROR'].rstrip('/'); \
        files=(m[:-len('simple')] if m.endswith('simple') else m+'/').rstrip('/'); \
        p=pathlib.Path('uv.lock'); t=p.read_text(); \
        p.write_text(t.replace('https://pypi.org/simple', m).replace('https://files.pythonhosted.org', files))"; \
    fi

# uv 缓存用 BuildKit cache mount：跨构建复用已下载的 wheel。
#
# **注意：本机 Docker 默认不是 BuildKit**，`--mount=type=cache` 会直接报
# "the --mount option requires BuildKit"。为了保证"任何环境一条命令能构建"，
# 这里不用 BuildKit 专属语法；分层本身已经保住了主要收益：
# 改源码只重装 5 个 workspace 包（editable，秒级）。
# 如已启用 BuildKit（DOCKER_BUILDKIT=1），可自行加 cache mount 进一步提速。
RUN uv sync --all-packages --no-dev --frozen --no-install-workspace

# ---- 源码层 ----
COPY packages ./packages
COPY apps/api ./apps/api
COPY apps/worker ./apps/worker

# 第二次 sync 只安装 workspace 包（editable 指向 /app/packages、/app/apps），
# 不重新解析也不重装第三方依赖。
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
#
# 注意：这里**不再需要** HuggingFace 缓存目录——embedding 已换成远程 API，
# 镜像也不包含 torch/sentence-transformers。
RUN mkdir -p /app/.data/storage \
    && chown -R rag:rag /app

USER rag

EXPOSE 8000

# 默认启动 API；Worker 与迁移在 compose 里用 command 覆盖。
CMD ["uvicorn", "--factory", "rag_api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
