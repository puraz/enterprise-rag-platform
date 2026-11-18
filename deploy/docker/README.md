# Docker 部署说明

> 适用：本机开发、演示、单机试运行。生产请把密码/密钥换成平台密钥管理，
> 并按需拆分 migration / admin / app 三个数据库角色。

## 0. 前置检查

```bash
docker --version
docker info            # daemon 必须已启动
docker compose version # 需要 Compose v2 插件
```

如果 `docker compose` 报 `unknown command`，安装 Compose 插件：

```bash
# Arch
sudo pacman -S docker-compose
# Debian/Ubuntu
sudo apt install docker-compose-plugin
# 或者独立二进制（不需要 sudo 的 rootless 场景）
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
```

启动 daemon：

```bash
sudo systemctl enable --now docker
# 或 Docker Desktop
```

## 1. 配置 Key

Compose **不会**把宿主 `.env` 整体注入容器，只读取下面几个变量做插值：

| 变量 | 必需 | 说明 |
|---|---|---|
| `RAG_LLM__API_KEY` | 是（否则问答 503） | DeepSeek / OpenAI 兼容 key |
| `RAG_EMBEDDING__API_KEY` | 是（否则文档停在向量化） | 默认 provider 是智谱 `embedding-3` |
| `RAG_RERANKER__API_KEY` | 是（否则重排降级为 RRF） | 默认 provider 是 SiliconFlow `/rerank` |
| `PYPI_MIRROR` / `PIP_INDEX_URL` | 国内建议 | 构建期 PyPI 镜像（不是应用配置） |
| `RAG_IDENTITY__JWT_SECRET` | 建议 | 不设则用镜像内的 dev 默认值 |
| `POSTGRES_PASSWORD` | 生产必须 | 默认 `rag_local_dev_only` |
| `RAG_APP_PASSWORD` | 生产必须 | 应用角色密码 |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | 生产必须 | 默认 `ragminio` / `ragminio123` |
| `GRAFANA_ADMIN_PASSWORD` | 可选 | 默认 `admin` |

仓库根的 `.env` 已被 compose 自动读取，本机开发通常只需要确认：

```bash
grep -E '^RAG_(LLM|EMBEDDING|RERANKER)__API_KEY=' .env
```

不要用 `RAG_ENV=prod` 跑这套 compose：生产守卫会拒绝开发密码、
dev 登录、HS256、以及缺失的 embedding key。

## 2. 启动

```bash
make up          # docker compose up -d --build
make ps          # 看状态
make logs        # 跟踪日志
```

访问：

| 入口 | 地址 |
|---|---|
| Web | http://localhost:8080 |
| API 文档 | http://localhost:8010/docs |
| 健康检查 | http://localhost:8010/health/ready |
| Prometheus（profile） | http://localhost:9090 |
| Grafana（profile） | http://localhost:3000 |
| Langfuse（profile） | http://localhost:3030 |

默认对象存储是本地卷 `appstorage`（`/app/.data/storage`），不启动 MinIO。
需要验证 S3 时：

```bash
docker compose -f compose.yaml -f deploy/docker/compose.s3.yml \
               --profile s3 up -d --build
# MinIO 控制台：http://localhost:9001
```

首次启动流程：

```
postgres 初始化（建 rag_app + pgvector）
  → migrate 跑 alembic upgrade head
  → api / worker 启动
  → web 启动
```

**没有本地模型要下载**：embedding/rerank 都是远程 API，启动即可用。
因此镜像不再包含 torch/CUDA，构建与启动都快得多。

> 换 embedding 模型/provider 后，旧向量属于另一个向量空间，**必须重建**：
> `make docker-reembed`。它走 A10 的零停机流程：
> 新建 building 版本 → 回填（检索仍用旧 ACTIVE）→ 校验条数 → 切换 ACTIVE。
> 超时/失败不会切换；旧版本置 retired，向量保留可回滚。
> 模型未变但想重建索引（调参/演练）加 `--force-new-version`；
> 只建版本不切换用 `--no-wait`。详见 ADR 0010。

登录：登录页输入 `dev:alice@example.com`，首次点「创建租户并登录」
（`/api/v1/auth/dev-bootstrap`）。这是本地开发专用端点，
生产关闭。

## 3. 常用命令

```bash
make down                 # 停止（保留数据卷）
make down-v               # 停止并删除数据卷（谨慎：清空库/文件/模型缓存）
make ps
make logs
make docker-migrate       # 单独跑迁移
make docker-backfill      # 回填 chunks.search_text（升级后执行一次）
make docker-reembed       # 换 embedding provider 后重算全部向量
make obs-up               # 起 Prometheus + Grafana
make obs-down
make lf-up                # 起 Langfuse 并开启追踪（宿主 :3030）
make lf-down
```

数据库：

```bash
# 宿主连容器 PG（端口刻意避开本机 5432）
psql -h 127.0.0.1 -p 5433 -U rag_admin -d rag

# 进入 API 容器
docker compose exec api bash
```

## 4. GPU 与本地模型

**已不再需要**。镜像不包含 torch/CUDA，embedding 走远程 OpenAI 兼容 API。
需要自建模型时（数据不能出境），任何提供 `/embeddings` 的服务都可以：

```yaml
services:
  worker:
    environment:
      RAG_EMBEDDING__PROVIDER: "openai_compatible"
      RAG_EMBEDDING__BASE_URL: "http://text-embeddings:8080/v1"
      RAG_EMBEDDING__MODEL_NAME: "BAAI/bge-m3"
      RAG_EMBEDDING__DIMENSION: "1024"
```

不需要重新引入 `sentence-transformers`/`torch`：TEI、vLLM 都自带
OpenAI 兼容端点。

## 5. 重排 / 自省

重排默认**开启**（SiliconFlow 的 Cohere 风格 `/rerank`）：

```yaml
RAG_RETRIEVAL__USE_RERANK: "true"
RAG_RERANKER__PROVIDER: "siliconflow"
RAG_RERANKER__MODEL_NAME: "Qwen/Qwen3-Reranker-4B"
RAG_RERANKER__API_KEY: "..."
```

换供应商（Cohere / Jina / 其它同形状端点）只改 `BASE_URL` 与 `MODEL_NAME`。
缺 key 或 provider=none 时，重排会**显式降级为 RRF 顺序**，
启动日志有对应 warning。

自省：`RAG_RETRIEVAL__USE_SELF_CHECK=true`，
`RAG_RETRIEVAL__SELF_CHECK_ACTION=flag|refuse`。

成本：embedding / rerank 的 token 已写入 `model_calls`（与 LLM 同口径，
含 trace_id）。单价默认不填（只记 token、成本 0），按合同设置：

```yaml
# 示例值（不是供应商报价！按你的合同填 USD / 1M tokens）
RAG_EMBEDDING__UNIT_PRICE_INPUT: "0.5"
RAG_RERANKER__UNIT_PRICE_INPUT: "0.5"
```

不填时历史记录仍可用后补单价重算（成本 = tokens × 单价）。

## 6. 可观测性

```bash
make obs-up   # Prometheus + Grafana
make lf-up    # Langfuse（并让 api/worker 开始上报 trace）
```

- Prometheus 同时抓取：
  - `api:8000/health/metrics`
  - `worker:9101/metrics`（worker 进程内的 metrics 守护线程）
- 告警规则复用 `deploy/observability/prometheus/alerts.yml`。
- Grafana 自动加载 `deploy/observability/grafana/dashboard.json`，
  datasource UID 固定为 `prometheus`，dashboard 用 `${DS_PROMETHEUS}` 变量引用。

### Langfuse

`--profile langfuse` 会起：`langfuse-web`（宿主 3030）、`langfuse-worker`、
ClickHouse、SeaweedFS（S3 兼容，替代 MinIO——本机镜像加速器拒绝 `minio/*`）。
元数据用业务 PG 的独立 database `langfuse`，不影响业务迁移。

首次启动自动创建组织/项目/账号，并**固定** API key：

| 项 | 值 |
|---|---|
| UI | http://localhost:3030 |
| 账号 | `admin@rag.local` / `raglangfuse123` |
| public key | `pk-lf-local-dev` |
| secret key | `sk-lf-local-dev` |

`make lf-keys` 可随时回看这些默认值。
追踪走 OpenTelemetry 上报（`/api/public/otel/v1/traces`），
启动时用 `auth_check` 主动探活：host/key 配错会在日志里明确告警，
而不是\"服务正常但一张图都没有\"。

## 7. 没内置什么，为什么

| 组件 | 状态 | 原因 |
|---|---|---|
| MinIO | 可选 `s3` profile | 默认本地存储避免镜像拉取成为启动阻塞；需要验证 S3 时显式开启 |
| Keycloak | 未内置 | 仓库目前没有 realm 导入配置；本地用 dev 登录。接 OIDC 时按 `RAG_OIDC__*` 配置真实 IdP 即可 |
| Langfuse | 可选 `langfuse` profile | v3 依赖 ClickHouse/Redis/S3，已随 profile 提供；默认不启动以控制资源占用 |
| Admin 前端 | 未内置 | 仓库当前只有 `apps/web`，没有 admin 应用，不虚构服务 |

## 8. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `docker compose` 不存在 | 装 Compose v2 插件（见第 0 节） |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker` |
| migrate 一直失败 | `make logs` 看是否连错库；确认 `RAG_APP_PASSWORD` 与 postgres 初始化时一致。**首次启动后改密码不会自动同步到已初始化的数据卷**，需 `make down-v` 重建 |
| api 启动即退出 | 检查 `RAG_DATABASE__*`；compose 已硬编码容器服务名，不要再注入宿主 `.env` 的整体内容 |
| 问答 503 | 没配 `RAG_LLM__API_KEY`；其它接口不受影响（惰性构造） |
| 文档一直停在「向量化」 | 检查 `RAG_EMBEDDING__API_KEY` / `BASE_URL` / `MODEL_NAME`；`make logs` 看 `embedding_*` 与 `task_failed` |
| 重排没生效 | `RAG_RERANKER__PROVIDER` 为 `none` 或缺 key 时会降级为 RRF；启动日志有 `rerank_*` warning |
| 检索结果相关性突然变差 | 换过 embedding 模型但没重算向量：`make docker-reembed` |
| 上传失败 | 默认使用本地卷；若开了 `s3` profile，检查 `minio-init` 与 bucket；S3 必须 path-style |
| 端口已被占用 | 宿主已有 PG(5432)/API(8000) 时，本栈用 5433/8010；Langfuse 用 3030（避开 Grafana 3000） |
| Langfuse 里没有数据 | 先 `make lf-up`（它会带上 `RAG_TRACING__ENABLED=true` 重建 api/worker）；再看启动日志的 `langfuse_auth_check_failed` |
| 构建卡在下载依赖 | 国内直连 pypi 很慢；在 `.env` 设 `PYPI_MIRROR` / `PIP_INDEX_URL`（默认已配阿里云镜像） |
