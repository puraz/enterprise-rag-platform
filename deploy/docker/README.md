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

## 1. 配置 LLM Key

Compose **不会**把宿主 `.env` 整体注入容器，只读取下面几个变量做插值：

| 变量 | 必需 | 说明 |
|---|---|---|
| `RAG_LLM__API_KEY` | 是（否则问答 503） | DeepSeek / OpenAI 兼容 key |
| `RAG_IDENTITY__JWT_SECRET` | 建议 | 不设则用镜像内的 dev 默认值 |
| `POSTGRES_PASSWORD` | 生产必须 | 默认 `rag_local_dev_only` |
| `RAG_APP_PASSWORD` | 生产必须 | 应用角色密码 |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | 生产必须 | 默认 `ragminio` / `ragminio123` |
| `GRAFANA_ADMIN_PASSWORD` | 可选 | 默认 `admin` |

仓库根的 `.env` 已被 compose 自动读取，所以本机开发通常只需要确认：

```bash
grep RAG_LLM__API_KEY .env
```

不要用 `RAG_ENV=prod` 跑这套 compose：生产守卫会拒绝开发密码、
dev 登录、HS256 等配置。

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

首次向量化会下载 `BAAI/bge-m3`（约 2.2GB）到 `hfcache` 卷，
之后复用。API 与 Worker 第一次用到时才加载模型，所以启动很快，
但第一次上传/问答会明显变慢。

> 镜像按 `uv.lock` 安装 torch（当前锁是 PyPI 默认的 CUDA wheel），
> 体积较大（数 GB）。CPU-only 机器仍可正常运行，只是构建/拉取更慢。
> 若要显著减小镜像，需要单独维护 CPU torch 的 lock（当前未做）。

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
make obs-up               # 起 Prometheus + Grafana
make obs-down
```

数据库：

```bash
# 宿主连容器 PG（端口刻意避开本机 5432）
psql -h 127.0.0.1 -p 5433 -U rag_admin -d rag

# 进入 API 容器
docker compose exec api bash
```

## 4. GPU

```bash
docker compose -f compose.yaml -f deploy/docker/compose.gpu.yml up -d --build
```

这会：

1. 给 api/worker 声明 NVIDIA 设备；
2. 把 `RAG_EMBEDDING__DEVICE` 设为 `cuda`。

镜像按 `uv.lock` 安装，torch wheel 已包含 CUDA 运行时，不需要额外换 lock。
显存预算按 **api + worker 各一份模型** 计算。

## 5. 打开重排 / 自省

改 `compose.yaml`（或写一个本地 override 文件，不要提交）：

```yaml
services:
  api:
    environment:
      RAG_RETRIEVAL__USE_RERANK: "true"
      RAG_RERANKER__PROVIDER: "local_bge"
      RAG_RERANKER__MODEL_NAME: "BAAI/bge-reranker-base"
  worker:
    environment:
      RAG_RETRIEVAL__USE_RERANK: "true"
      RAG_RERANKER__PROVIDER: "local_bge"
```

自省同理：`RAG_RETRIEVAL__USE_SELF_CHECK=true`，
`RAG_RETRIEVAL__SELF_CHECK_ACTION=flag|refuse`。

## 6. 可观测性

```bash
make obs-up
```

- Prometheus 同时抓取：
  - `api:8000/health/metrics`
  - `worker:9101/metrics`（worker 进程内的 metrics 守护线程）
- 告警规则复用 `deploy/observability/prometheus/alerts.yml`。
- Grafana 自动加载 `deploy/observability/grafana/dashboard.json`，
  datasource UID 固定为 `prometheus`，dashboard 用 `${DS_PROMETHEUS}` 变量引用。

## 7. 没内置什么，为什么

| 组件 | 状态 | 原因 |
|---|---|---|
| MinIO | 可选 `s3` profile | 默认本地存储避免镜像拉取成为启动阻塞；需要验证 S3 时显式开启 |
| Keycloak | 未内置 | 仓库目前没有 realm 导入配置；本地用 dev 登录。接 OIDC 时按 `RAG_OIDC__*` 配置真实 IdP 即可 |
| Langfuse | 未内置 | v3 依赖 ClickHouse 等一整套组件，单机 compose 会显著变重；先用 Prometheus + 结构化日志 |
| Admin 前端 | 未内置 | 仓库当前只有 `apps/web`，没有 admin 应用，不虚构服务 |

## 8. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `docker compose` 不存在 | 装 Compose v2 插件（见第 0 节） |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker` |
| migrate 一直失败 | `make logs` 看是否连错库；确认 `RAG_APP_PASSWORD` 与 postgres 初始化时一致。**首次启动后改密码不会自动同步到已初始化的数据卷**，需 `make down-v` 重建 |
| api 启动即退出 | 检查 `RAG_DATABASE__*`；compose 已硬编码容器服务名，不要再注入宿主 `.env` 的整体内容 |
| 问答 503 | 没配 `RAG_LLM__API_KEY`；其它接口不受影响（惰性构造） |
| 上传失败 | 默认使用本地卷；若开了 `s3` profile，检查 `minio-init` 与 bucket；S3 必须 path-style |
| 端口已被占用 | 宿主已有 PG(5432)/API(8000) 时，本栈用 5433/8010；停掉本地 API 或改 compose 端口映射 |
| 第一次问答很慢 | 在下载/加载 bge-m3（约 2.2GB）；看 `make logs` 的 `embedding_model_loaded` |
| OOM | api/worker 各加载一份 bge-m3，建议 ≥8GB 内存；也可先只起 api 做问答验证 |
