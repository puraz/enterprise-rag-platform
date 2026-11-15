# AGENTS.md

> 教学方式/带练节奏不在这里，那是 docs/PROMPT.md。
> 最后更新：2026-11-14

## 1. 项目是什么（一句话 + 边界）
企业级多租户 RAG 平台，4 端 + 1 契约的 Monorepo：
- apps/api        FastAPI 接口端（用户面 /api/v1 + 管理面 /api/v1/admin）
- apps/worker     异步管线、索引重建、定时任务
- apps/web        React 用户前端
- apps/admin      React 后台管理端
- packages/contracts  四端共享契约（唯一真相来源）
- packages/core       横切基础设施（配置/日志/异常/DB/缓存/韧性），api 与 worker 共用
不是什么：不是 demo、不是单脚本、不是 Jupyter 实验。

## 2. 怎么跑起来（照抄能跑的命令）
- 环境准备：`uv python install 3.12`（首次一次即可）
- 装依赖：  `uv sync --all-packages`（仓库根执行；**必须带 --all-packages**，
             否则只装根工程，workspace 包不进 .venv，import 会 ModuleNotFoundError）
- 依赖自检：`make check-deps`（验证 PG / Redis / pgvector 是否就绪）
- 后端：    `uv run --package rag-api uvicorn rag_api.main:app --reload --port 8000`
- 起依赖：  **本机模式**（当前开发机，无 Docker）：直接用系统 PG(5432) + Valkey(6379)，
             连接参数见 `.env`；扩展由首个迁移创建。
             **Docker 模式**（换机器）：`make up`（PG/Redis/MinIO/Keycloak/Langfuse）
- 迁移：    `make migrate` / `make migrate-new name=xxx` / `make migrate-down`
- 测试：    `make test`（单测，**无需外部依赖**）、`make test-int`（集成测，连真实 PG/Redis）、
             `make test-all`（全量）、`make test-concurrency`
- 前端：    `cd apps/web && pnpm i && pnpm dev`
- 类型/规范：`make lint`（ruff+mypy+eslint）、`make typecheck`
- 契约重新生成：`make gen-contracts`（改了 Pydantic schema 后必跑）
> 改完后端 schema 不跑 gen-contracts，前端类型就会漂移——这是最常见的错。

## 3. 目录与分层（东西该放哪）
Python 包（uv workspace，src/ 布局）：
- packages/core        `rag_core`        横切基础设施（配置/日志/异常/DB池/Redis/韧性/PII），不含业务
- packages/contracts   `rag_contracts`   Pydantic 契约 + 枚举 + 错误码 + 队列消息（叶子包，零内部依赖）
- packages/db          `rag_db`          ORM 模型 + 仓储(SQL) + Alembic 迁移
- packages/adapters    `rag_adapters`    可插拔适配层：文档解析 / 对象存储 / 模型 Provider
- apps/api             `rag_api`         路由/服务/依赖注入
- apps/worker          `rag_worker`      管线/定时任务

**为什么有 packages/db（不能省略，也不能塞进别处）**：
ORM 模型与仓储必须被 api 与 worker **同时**使用（worker 要改任务状态、写 chunk、
读文档元数据），但「worker 禁止 import apps/api」是硬约束，而 core 又规定
「不含任何业务概念」。三条约束叠加后，模型与仓储只能落在第三个共享包里。

**为什么有 packages/adapters（同一个矛盾的第二次出现）**：
解析器（PyMuPDF）、存储后端（boto3）、Embedding（bge-m3）原本写在 `rag_api` 里，
但 worker 才是它们的主要消费者（所有解析/向量化都在 worker 里跑）。
两条硬约束再次交叉 → 只能下沉成共享包。
判断标准：**api 与 worker 都要用，且不认识 ORM / 路由 / 请求上下文**。
适配层**不得 import `rag_db`**：它只做协议转换，不读业务表；
"查 blob 地址再下载"这类编排留在调用方（worker handler / service）。

依赖方向（严格单向，`tests/test_workspace.py` 会检查）：
```
contracts ← core ← db ← api / worker
                 ↑
              adapters ← api / worker
```
api 与 worker **互不依赖**，只通过契约、数据库状态、队列消息通信。

后端分层（apps/api 内）：api(路由) / schemas / services(业务) / deps(依赖注入) /
          pipelines / retrievers / chunkers / identity / evaluators / workers / openapi
（parsers / storages / providers 已迁出到 packages/adapters）
前端分层：routes / features / components / hooks / api(自动生成，别手改) / stores
- 路由里不写业务逻辑；service 里不写 SQL；SQL 只在 packages/db 的 repository。
- 依赖方向单向向内，禁止反向 import。

**依赖注入按用途分包**（`rag_api/deps/`），不是一个 deps.py 堆到底：
| 文件 | 职责 |
|---|---|
| `app.py` | 进程级单例（Database/Redis/TokenService/Settings），从 app.state 取 |
| `auth.py` | `Principal`、`get_current_principal`（只验签不查库）、`require_platform_role` |
| `database.py` | `get_db`（带租户上下文）/ `get_session`（无上下文）/ `get_admin_db`（跨租户） |
| `pagination.py` | 分页参数统一解析与上限校验 |
业务代码统一 `from rag_api.deps import get_db`，不直接 import 内部模块。

## 4. 硬约束（违反即打回，没有例外）
- **端边界**：跨端只走 API 或队列消息。apps/worker 禁止 import apps/api；
  共享代码只能进 packages/contracts、packages/core、packages/db 或 packages/adapters。
- **可插拔**：换 LLM/向量库/存储/IdP/队列/评估器，只应动一个适配层文件 + 一行配置。
  见到 `if provider == "xxx"` 一律改成能力匹配或策略注册。

### 4.4 模型 Provider 层规范（`apps/api/src/rag_api/providers/`）

- **LLM 用 `langchain-openai`**（`langchain_llm.py`），不自研 HTTP 客户端。
  理由：LangChain 的 `InputTokenDetails.cache_read` / `OutputTokenDetails.reasoning`
  完整覆盖成本核算；SSE 流式解析成熟；A11 的 LangGraph 原生集成。
  **不要以"框架会丢字段"为理由自研** —— 先实测再判断。
- **必须 `max_retries=0`**：LangChain 默认重试 2 次，不关会与 `ResilientCaller`
  形成乘数效应，且熔断器拿不到真实失败信号。
- **每个 Provider 用独立 `httpx.AsyncClient`**：openai SDK 默认共享模块级
  client，某个实例 `aclose()` 会让其它实例抛 `Event loop is closed`。
- **LLM 与 Embedding 配置分开**（`LLMSettings` / `EmbeddingSettings`）：
  换 LLM 模型不应触发 embedding 重建。
- **Embedding 维度必须在工厂校验**：`settings.embedding.dimension` 必须等于
  `LocalBGEEmbedding.capabilities.dimension` 且等于 `chunks.embedding` 的
  `vector(1024)`。三处不一致会在入库时才炸，届时已切分完几百个 chunk。
- **Embedding 必须归一化**（`normalize_embeddings=True`）：余弦相似度 = 内积，
  计算更快且与 `vector_cosine_ops` 索引语义一致。
- **CPU 密集操作必须切线程**：`model.encode()` 和模型加载都是同步阻塞的，
  直接 `await` 会阻塞整个事件循环。
- **批量顺序必须与输入严格对应**：错位会让"检索能跑但相关性全乱"，且不报错。
- **Reranker 在 A11 实现**：A5 用 `NoOpReranker` 占位（对应降级链
  "Reranker 挂 → 只用 RRF"）；rerank 的价值只能在真实检索结果上验证。
- **抽象优先**：所有外部能力面向 ABC/Protocol + 工厂 + 配置注入，业务不 import 具体实现。
- **韧性统一**：重试/熔断/超时/信号量只能走 ResilientCaller，禁止业务自己写重试循环。
- **鉴权服务端注入**：tenant_id、ACL 过滤从 token 解析，绝不取请求参数；RLS 是最后一道防线。
- **并发安全**：状态流转用乐观锁(version)；计数走 Redis 原子自增+对账；
  唯一性靠 DB 唯一索引，不靠"先查再插"；跨存储一致性用 outbox，不写跨库大事务。
- **密钥**：只走环境变量 RAG_*，不进库明文、不进日志、不进 trace；Langfuse 打点对
  Authorization 头脱敏。
- 全异步 I/O；CPU 密集(解析/OCR/embedding)隔离到进程池或 worker，不阻塞事件循环。
- mypy 严格 + ruff + TS strict + eslint 必须通过；全量类型注解；中文注释。

### 4.1 响应契约（前端对接的唯一形状）

#### 成功响应：**无 `code`、无 `data` 包裹**

```json
// 详情 / 创建 / 更新
200 {"id": "...", "title": "制度.pdf", "created_at": "2026-11-15T05:17:08Z"}

// 列表
200 {"items": [...], "meta": {"limit": 20, "offset": 0, "total": null, "has_more": true}}

// 删除
204 (空 body)
```

- **成功不带 `code`**：`code` 是失败原因，成功没有原因；HTTP 2xx 已表达成功。
- **不包 `data` 层**：那是"成败都用 200"方案才需要的，用状态码区分成败时纯属噪音。
- 只有三类形态：**单对象** / **分页集合** / **无内容**。不创造第四种。

#### 状态码

| 场景 | 状态码 |
|---|---|
| GET 详情/列表、PATCH/PUT | 200 |
| POST 创建资源 | **201** + `Location` 头 + 资源对象 |
| POST 触发异步任务 | **202** + `{task_id, poll_url}` |
| DELETE | **204**（无 body） |

#### 字段命名硬规则

| 规则 | 正确 | 禁止 |
|---|---|---|
| 全部 `snake_case` | `created_at` | `createdAt` |
| 主键 `id` | `{"id": ...}` | `document_id` / `docId` |
| 时间 `*_at` + ISO 8601 带时区 | `"2026-11-15T05:17:08Z"` | 时间戳 / 无时区 |
| 列表 `items` | `{"items": [...]}` | `data` / `list` / `results` |
| 布尔 `is_*`/`has_*`/`can_*` | `is_current` | `current` |
| 计数 `*_count`（整数） | `chunk_count` | `chunks` |
| 枚举值小写 snake_case | `"in_progress"` | `"InProgress"` |

`tests/test_success_contract.py` 会检查所有这些规则。

#### 失败响应：唯一形状 `ErrorResponse`

`{code: int, code_name: str, message, request_id, fields, details}`。
FastAPI 默认的 `{"detail": ...}` 已由全局处理器转换，不得绕过。
- **每个路由必须声明 `responses=`**，否则 OpenAPI 里只有 200，
  前端生成 TS 时看不到错误类型。用 `rag_api/openapi.py` 的现成组合：
  `PUBLIC_ENDPOINT` / `AUTHENTICATED_ENDPOINT` / `TENANT_ENDPOINT` /
  `TENANT_WRITE_ENDPOINT` / `ADMIN_ENDPOINT` / `UPLOAD_ENDPOINT`。
  `tests/test_openapi_contract.py` 会检查所有路由都有非 2xx 声明，
  且错误 schema 必须是 `ErrorResponse`（健康检查例外）。
- **状态码语义**：创建返回 `201`、无返回体用 `204`、异步任务已接受用 `202`；
  不在业务异常里手写状态码，`AppError.http_status` 由 `category` 推导
  （**不要按码的域号段推导**：0xxxx 通用域里既有 422 又有 404）。
- **401 只用于身份不成立**，403 只用于身份成立但无权，
  越权访问资源统一返回 404（不泄漏资源存在性）。

### 4.1.1 业务错误码（双轨：数字码 + 语义名）

- `code` 是**数字**（`IntEnum`），前端按它 `switch`、按号段批量判断、写告警规则。
- `code_name` 是**语义名**，日志/审计/排障用它（`10101` 在日志里不可读）。
- **HTTP 状态码与业务码解耦**：同一业务码在不同路由可能有不同状态码，
  但业务码永不改变语义。
- **号段规划**（每域 10000 个，新增域从号段表申请，不重排已有码）：

  | 号段 | 域 |
  |---|---|
  | `0xxxx` | 通用/框架（校验、路由、限流） |
  | `1xxxx` | 认证 AuthN（token、登录、IdP） |
  | `2xxxx` | 授权 AuthZ（角色、ACL、密级、跨租户） |
  | `3xxxx` | 知识库与文档 |
  | `4xxxx` | 分块/索引/检索 |
  | `5xxxx` | 上传与存储 |
  | `6xxxx` | 任务与队列 |
  | `7xxxx` | 配额与计费 |
  | `8xxxx` | 外部供应商 |
  | `9xxxx` | 内部系统 |

- **码一旦发布不可改**：前端会硬编码。需要新语义时新增码、
  把旧码标 deprecated，**绝不复用已废弃的码**。
- 新增错误码必须同时更新：`rag_contracts.errors.codes.ErrorCode`、
  对应异常类（`rag_core.errors`）、必要时 `_CODE_STATUS_OVERRIDES`。
  `tests/test_error_codes.py` 检查号段合法性、唯一性、名字反查一致性。

### 4.2 分页契约（已放宽"禁用 OFFSET"）

- 列表接口统一返回 `Page[T]`：`{items: [...], meta: {limit, offset, total, has_more}}`。
- 分页参数走 `rag_api.deps.page_query`（`limit` 上限 100、`offset` 上限 10000）。
  `offset` 超限返回 400，引导前端用过滤条件收窄范围，不让深翻页扫全表。
- **`total` 默认不返回**（`include_total=false`）：`COUNT(*)` 在大表上很慢。
  管理端需要"共 N 条"时才显式请求；列表默认只依赖 `has_more` 做无限滚动。
- 深翻页的大表（审计日志）单独提供游标接口，不复用本契约。

### 4.3 OpenAPI 输出

- 契约真相是 `packages/contracts` 的 Pydantic 模型；OpenAPI 由 FastAPI 生成，
  再由 `make gen-contracts` 转成前端 TS（Pydantic → JSON Schema → TS）。
- **改了响应模型必须重跑 `make gen-contracts`**，`make check` 会拦住漂移。

## 5. 禁区（never do）
- 不手改 packages/contracts 里自动生成的前端类型（改源头 schema 再 gen）。
- 不把 API Key、真实租户数据、用户文档内容写进日志/异常/trace/测试 fixture。
- 不新增同步阻塞调用到事件循环里（含 boto3 直调，要放线程池）。
- 分页统一走 `Page[T]` + `page_query` 依赖（见 4.2）；`offset` 上限 10000，
  超限返回 400 而不是静默扫全表。**大数据量的深翻页场景必须用游标**。
- 不写 `SELECT *`；不在循环里查库(N+1)；列表查询必须显式 `limit`。
- 不在管理面之外暴露管理接口；高危操作不加审计与二次确认。
- 不编造供应商模型名/API/配置项——不确定就去官方文档核对或问我。

## 6. 命名约定
- Python 包：`rag_core` / `rag_contracts` / `rag_db` / `rag_api` / `rag_worker`（PEP 8 全小写下划线）
- 发行包名：`rag-core` / `rag-contracts` / `rag-db` / `rag-api` / `rag-worker`
- 配置类：`Settings`，子配置按域拆 `DatabaseSettings` / `RedisSettings` / `LLMSettings`
- 枚举：`packages/contracts/src/rag_contracts/enums/` **按业务域分包**
  （`tenancy.py` / `documents.py` / `tasks.py` …），一个域文件对应一组模型；
  **禁止堆成单文件**。新增枚举后必须登记到 `export_schema.py` 的 `EXPORTED_ENUMS`，
  否则前端拿不到类型（`tests/test_enums.py` 会自动扫描校验）。
  错误码是例外，在 `rag_contracts.errors` 且值用 `DOMAIN.REASON` 大写格式。
- 表名：`snake_case` 复数（`documents` / `chunks` / `audit_logs`）
- 前端：组件 PascalCase 文件名，hook 用 `useXxx.ts`，其余 kebab-case

## 7. 验证（改完怎么证明没坏）
- 单测（`make test`，零外部依赖）+ 集成测（`make test-int`，连真实 PG/Redis）+ 并发专项 + 覆盖率≥70%
> 当前开发机无 Docker，集成测直连本机 PG/Valkey；换到有 Docker 的环境再用 testcontainers，
> 测试代码不为此写两套——连接信息统一从 `Settings` 读。
- 检索改动：用 tests/eval 黄金集跑前后对比，看 recall/MRR/引用准确率，不凭感觉
- 压测后对账：条数一致、无重复、无越权、成本账平
- 前端 playwright 覆盖登录/对话/上传主流程

## 8. 约定与规范细节
- 命名要规范，包含文件名，变量名等等
- 提交：conventional commits（feat/fix/chore/docs/refactor/test），中文描述
- 架构决策记 docs/adr/，进度记 docs/PROMPT-PROGRESS.md，选型结论记 docs/PROMPT-DECISIONS.md
- **数据库变更只走 Alembic 迁移**：禁止手工 `ALTER TABLE`、禁止改完模型不生成迁移。
  迁移必须走 expand-contract（先加列/回填/再删列），保证滚动发布期间新旧代码都能跑。
- 教学/带练方式见 docs/PROMPT.md（本文件不管教学）

### 4.5 对象存储层规范（`apps/api/src/rag_api/storages/`）

- **一套代码兼容多家**：S3 协议被 AWS/MinIO/OSS/COS/七牛兼容，
  换厂商只改 `RAG_STORAGE__S3_ENDPOINT_URL` 与凭证，代码零改动。
- **boto3 是同步库，必须 `asyncio.to_thread`**：直接调用阻塞事件循环。
  不要用 aioboto3（连接池与 session 生命周期不一致，长时间运行会泄漏）。
- **`StorageBackend.get` 声明为普通 `def` 而非 `async def`**：它返回异步迭代器，
  本身不是协程。写成 async def 会让调用方得到 `Coroutine`，
  `async for` 报 "not async iterable"（mypy 能抓到）。
- **大文件必须流式**：`put` 接受异步迭代器逐块落盘；`get` 返回惰性流。
  2GB 文件整块入内存会在并发几个请求时 OOM。
- **写入用「临时文件 + 原子重命名」**：直接写目标路径中途失败会留下
  半个文件，而 `head` 会认为对象完整——静默数据损坏。
- **分片编号从 1 开始**（S3 语义）。0-indexed 与 1-indexed 混用是
  分片上传最常见的 bug，且错误发生在所有分片传完之后。
- **合并必须按 part_number 排序**：客户端并发上传到达顺序随机，
  不排序会得到内容错乱但大小正确的文件。
- **分片大小必须 ≥ 5MB（S3 硬要求）**：小于会在 complete 时报
  `EntityTooSmall`——最浪费的失败点。配置层已用 validator 拦截。
- **`list_parts` 必须处理分页**：S3 每次最多 1000 片，大文件不翻页会漏。
- **断点续传的权威来源是存储层**，不是数据库计数：
  `upload_sessions.uploaded_parts` 只是缓存，可能滞后。
- **秒传必须同租户内**：跨租户秒传是存在性侧信道
  （通过"秒传成功/失败"探测别的租户有没有某个文件）。
- **合并后必须重算整体 hash 并校验**：分片级 ETag 只能证明"这片传对了"，
  不能证明"整体与客户端原始文件一致"（分片顺序错、客户端计算错都会漏）。
- **`abort_multipart` 必须调用**：S3 不自动清理未完成的分片上传，
  会一直占用存储并计费。
- **存储键不含原始文件名**：文件名可能含隐私（"张三的体检报告.pdf"），
  且重命名文档不该改变存储位置。用 `{tenant}/{hash[:2]}/{hash}{suffix}`。

### 4.6 上传接口规范（`/api/v1/uploads`）

- **分片请求体是原始二进制**（`Content-Type: application/octet-stream`），
  不是 multipart：multipart 的 boundary 与头部会给每片加几十到几百字节，
  1000 片就白传几百 KB；且解析器会缓冲整个请求，大分片时内存翻倍。
- **分片用 PUT 而非 POST**：PUT 幂等，网络抖动时客户端可安全重试。
- **URL 里的 `upload_id` 是会话主键（UUID），不是存储层 upload_id**：
  后者是后端实现细节，暴露它会让攻击者推断存储配置。
- **分片编号从 1 开始**，越界返回 400。
- **除最后一片外，每片大小必须等于 `part_size`**（服务端校验）。
- **魔数校验分两阶段**：
  - `initiate` 时只有文件名与扩展名可校验（`head=None` 表示"还没有内容"）；
  - `complete` 时读到完整文件后必须校验魔数与扩展名一致，
    不匹配则删除对象并返回 415。
- **整体 hash 校验在 complete**：不匹配则删除对象 + 标记失败，
  绝不入库。
- **秒传只在同租户内**（`tenant_id + content_hash`）。
- **跨租户访问会话返回 404 而非 403**：不泄漏"这个 ID 是否存在"。
- **上传安全校验清单**（`services/upload_security.py`）：
  文件名净化（去路径/控制字符/保留字符、保留扩展名截断）、
  扩展名白名单（拒绝 .exe/.sh/.bat 等）、魔数校验、
  zip bomb 三维防护（条目数/压缩比/单条目大小）、zip slip 路径检查。

### 4.7 认证与 OIDC 规范（`apps/api/src/rag_api/identity/`）

- **不手写 JWT 验签**：用 PyJWT + 显式算法白名单 + 完整 claims 校验
  （iss / aud / exp / iat / sub）。
- **算法白名单绝不含 `HS256` 与 `none`**：HS256 需 IdP 与 RP 共享对称密钥
  （OIDC 里不存在）；`none` 等于不签名。两者都导致 token 可伪造。
  先校验算法再取公钥，**顺序反了等于让攻击者决定用什么算法验签**。
- **JWKS 按 kid 缓存 + kid 未命中时强制刷新一次**：IdP 轮换密钥后
  新 token 用新 kid；不刷新会导致所有用户突然登录失败，且根因不可见。
- **discovery 的 issuer 必须与配置一致**：防 discovery 文档被篡改指向别的 IdP。
- **SPA 用 Authorization Code + PKCE（S256）**：public client 存不住 secret，
  没有 PKCE 时授权码泄漏即可被换走。`code_verifier` 只在前端保存。
- **refresh token 必须轮换 + 复用检测**：
  - 每次续期作废旧 token 并签发新的（同 family）；
  - **检测到已用 token 再次出现 → 作废整族**（无法分辨攻击者与受害者，
    安全的选择是双方都重新登录）；
  - Redis 里**只存 SHA-256 hash**，不存原文（存储泄漏时无法直接使用）。
- **用户级失效检查 `tokens_valid_after`**：禁用账号或强制下线后，
  早于该时刻签发的 token 全部失效。只验签不看账号状态，
  会让被禁用者继续用满 access token 有效期。
- **不是所有 IdP 都是标准 OIDC**：企业微信/飞书/钉钉的 OAuth2 缺标准
  id_token 或 claims 不规范，需要单独适配器（`IdentityProvider` 接口相同）。
- **本机无 Docker/Java 时用 `rag_api.testing.MockOIDCServer` 验证代码逻辑**：
  它能覆盖验签、claims 校验、JWKS 缓存、PKCE、算法攻击。
  真实 Keycloak 的价值是验证部署 quirks（realm 路径、issuer 尾部斜杠）。

### 4.8 RBAC / ACL / 密级规范（`apps/api/src/rag_api/permissions/`）

- **两层分工**：RLS 防跨租户，ACL 防同租户内越权。
  缺任何一层都有洞：只有 RLS 时实习生能看到全租户文档；
  只有 ACL 时一旦 SQL 漏写条件就跨租户泄漏。
- **ACL 过滤必须在 SQL 的 WHERE 里**，不能查出来再在 Python 里过滤：
  - 分页错乱（limit 20 查出 20 条、过滤后剩 3 条）；
  - 越权内容进应用内存（日志/异常可能带出标题）；
  - 浪费 99% 的 I/O。
- **密级比较不能用字符串序**：`confidential` < `internal` < `public` < `secret`
  是按字母排的，与敏感程度无关。必须用 `LEVEL_RANK` 映射 + SQL `CASE WHEN`。
  直接字符串比较会让 `secret` 被当成最低级——权限完全错但不报错。
- **多角色取最低密级上限**：owner + viewer 按 viewer 算。
  按最高算会让"临时限制某人权限"变得必须移除全部角色。
- **不可读时返回 404 而非 403**（不泄漏资源存在性）。
- **未知密级按 SECRET 处理**（安全默认）：不认识的值不该被当成公开。
- **检索是最危险的路径**：chunk 内容是文档正文本身，不过滤等于用户
  能通过提问把无权文档"问出来"。
- **生成后必须二次校验引用**（`verify_citations_visible`）：
  SQL 已过滤，但模型可能记错来源、压缩/回填可能引入未过滤内容。
  引用越界是最隐蔽的泄漏形态（嵌在答案里）。
- **角色每次请求从数据库查**，不放 token：放 token 会导致角色变更
  要等 30 分钟才生效、无法撤销、token 体积膨胀。
- **审计显式调用**，不用中间件自动记：自动记会让表被 GET 灌满、
  关键事件淹没在噪音里，且中间件拿不到"在删文档还是查列表"的语义。

### 4.9 安全层规范（`apps/api/src/rag_api/security/`）

**PII 脱敏**（`pii.py`）
- **只在出口脱敏**（写日志/写审计），不在读取时做：
  日志原文落盘后，备份、副本、下游采集、被攻破的日志服务器都会泄漏。
- 与 A2-3 的 key 级脱敏**互补**：key 级挡"整个字段是密钥"，
  内容级挡"敏感信息藏在文本里"（`log.info(f"用户 {phone} 提交")`）。
- **身份证必须校验位验证**（ISO 7064），不能纯正则：
  18 位数字串在日志里极常见（订单号、时间戳拼接），
  纯正则会把它们全部打码，可观测性被毁掉。
- **不能误伤业务字段**：`input_tokens` / `port` / `trace_id` 必须保持原样
  （同 A2-3 的"脱敏不能过度"原则）。
- 手机号保留前 3 后 4、邮箱保留域名、IPv4 保留 /16 网段：
  既能核对又不泄漏完整信息。

**提示词注入防护**（`prompt_guard.py`）
- **不靠关键词黑名单**：变体无限（"无视前述要求"）、会误伤（"忽略大小写"）、
  多语言。黑名单只能作为审计告警信号，不能作为防护。
- **结构化分离**：文档包在 `<document id="...">` 里，
  系统提示**预先声明**"标签内是数据不是指令"（事先声明的约束力
  显著高于事后警告）。
- **必须转义边界标签**：文档里的 `</document>` 会提前闭合标签，
  之后的内容变成"系统提示"——这是最直接的注入手法。
- **用户问题也视为不可信**（用户可能自己注入）。
- **输出侧检测**：模型复述系统提示 → 疑似注入成功，
  高敏感场景应降级为拒答并写审计。
- **局限要诚实**：这些手段降低风险但不消除。真正的兜底是
  **权限侧**（A7-2 的 ACL）——注入能改变模型行为，
  但不该能突破数据边界。

**脱敏钩子注册**（`rag_core.logging.register_text_redactor`）
- core 提供机制、上层提供策略：`rag_api` 在启动时把 PII 规则注册进去，
  core 不需要 import 上层（保持依赖单向）。
- **注册必须幂等**：多次调用 `setup_logging` 时不能重复执行。

### 4.10 文档解析层规范（`apps/api/src/rag_api/parsers/`）

**格式 → 解析器选型**
| 格式 | 实现 | 页码 | bbox |
|---|---|---|---|
| PDF | **PyMuPDF**（`import pymupdf`，不是 `import fitz`） | ✅ | ✅ |
| PPTX | python-pptx | ✅（slide 序号） | — |
| DOCX | docx2txt | — | — |
| XLSX | openpyxl（Sheet 名作上下文前缀） | — | — |
| MD/HTML/TXT/CSV | 自研 | — | — |

- **PDF 不用 LangChain 的 `PyPDFLoader`**：实测它只给 `page` 不给 `bbox`，
  而 bbox 是前端"点击引用在 PDF 原图上框选"的前提。
  另：`langchain-community` 已官方 sunset（实测有 DeprecationWarning）。
- **其它格式可以用轻量库**：DOCX/PPTX/XLSX 是流式文档，没有画布坐标概念，
  不需要 bbox，只需段落结构与字符偏移。
- **必须用 `import pymupdf` 而非 `import fitz`**：`fitz` 是历史导入名，
  已弃用（实测有 FutureWarning），未来版本会移除。
- **字符偏移必须能精确切出 block.text**：`verify_char_offsets()` 校验，
  解析器的单测必须调它。偏移算错的后果是引用高亮显示错误文字，
  且端到端测试极难发现（答案对、高亮错）。
- **页码从 1 开始**：`pages[i].page_no` 是用户看到的页码，
  不是 0-based 索引。存错会让所有引用偏一页。
- **能力声明必须诚实**：只有真能给 bbox 的才声明 `supports_bbox=True`。
  虚假声明会让前端拿到 null，表现为"高亮时好时坏"。
- **块之间插入的分隔符也要计入偏移**：`\n\n` 是 `text` 的一部分。
  先拼 text 再按内容搜索定位会在重复文本上出错。
- **文本编码按 UTF-8 → GBK → gb18030 依次尝试**：中文文档常见 GBK，
  只试 UTF-8 会抛 UnicodeDecodeError 且看不出是编码问题。
- **CRLF 必须规范化为 LF**：否则偏移在 Windows 文件上错位。
- **OCR 是可选能力**：`build_ocr_engine()` 检测不到 tesseract 时返回
  `NullOcr`，调用方降级；**启动日志必须告警并给出安装提示**，
  否则"扫描件解析不出内容"会变成没人知道原因的谜。

### 4.11 Worker 与异步任务规范（`apps/worker/src/rag_worker/`）

- **数据库是任务的权威来源，队列只是通知**：API 写文档+任务+outbox 同事务，
  outbox 转发到队列让 worker 立刻处理，**同时 worker 每 2 秒轮询 DB 兜底**。
  只靠队列 → Redis 丢数据则任务永久丢失；只靠轮询 → 延迟至少一个周期。
- **认领必须跨租户，执行必须在租户内**（两套会话，不能混）：
  | 用途 | 连接 | 原因 |
  |---|---|---|
  | claim / reclaim / 进度 / 完成 / 失败 | **admin（BYPASSRLS）** | 必须看到全部租户的队列 |
  | handler 里的业务读写 | **租户会话（set_config）** | RLS 必须生效 |

  `tasks` 在 `RLS_TABLES` 里，用普通会话扫会被过滤成空——**不报错，只是永远
  捡不到任务**。反过来让业务读写也走 admin，等于把 RLS 整个撤掉。
- **进度回写必须用独立短事务**：写进工作事务里，提交前谁都看不到，
  前端进度会一直停在 0 然后跳到 100。`ProgressReporter` 自带会话工厂。
- **`progress=100` 只由 `complete()` 设置**，handler 上报上限 99。
  否则仍在跑（甚至马上要失败）的任务会显示"已完成"。
- **业务提交与"标记任务成功"分两个事务**：合成大事务会让事务持续
  整个任务时长（可能 10 分钟），占住连接、阻塞 vacuum、崩溃时白算。
  代价是两步之间崩溃会重派 → **handler 必须幂等**（不是建议是必需）。
  幂等做法：确定性存储键 + 乐观锁 / upsert。
- **失败分类决定重试**：RETRYABLE 才重试（指数退避，上限 600s），
  PERMANENT（文件损坏/payload 非法）与 FATAL（未知 bug/配置错）直接死信。
  把永久失败重试 3 次是纯浪费：白占 3 个槽位几分钟，用户多等三轮退避。
- **适配层异常靠 `retryable` 协议属性分类**，不用 isinstance：
  core 不能 import adapters。`StorageError(retryable=True)` 若被当成 FATAL，
  偶发网络抖动就变成永久失败。
- **回收崩溃任务不能省**（`reclaim_stale`）：worker 被 kill 后任务永远停在
  running，表现为"文档一直处理中而队列是空的"——没有报错，最难排查。
  回收时 `retry_count + 1`，防毒任务被无限重派。
- **SIGTERM 只置标志，不强杀在途任务**：强杀留下半成品数据，
  且任务要等 10 分钟 lock_timeout 才回收。
- **解析产物不进数据库**：全文写对象存储派生键
  `parsed/{tenant}/{version}.txt`（确定性键，重跑不产生孤儿），
  DB 只存 key + hash + 统计。全文进 JSONB 会撑大 TOAST、备份暴涨、
  切分时整块反序列化进内存。
- **取 ORM 关系必须 eager load**（`selectinload`）：async 会话里隐式懒加载
  直接抛 `MissingGreenlet`，不是"慢一点"。

### 4.12 分块层规范（`packages/adapters/src/rag_adapters/chunkers/`）

- **位置信息是引用高亮的前提**：chunk 必须带 `char_start`/`char_end`/`page_no`/`bbox`/`section_path`。
  丢了 `page_no`/`bbox` 等于永久失去"点击引用跳原文"的能力，而且后续环节无法补回。
- **解析产物必须是结构化的**：worker 写 `parsed/{tenant}/{version}.jsonl`
  （文本 + 块列表 + 页列表 + 偏移 + bbox + 标题层级），**不能只存纯文本**。
  纯文本里没有页码与坐标，切分时无从推断。同目录还写一份 `.txt` 供人工比对。
- **产物带 `format_version`**：跨进程/跨部署传递，改字段必须递增版本并在读侧拒绝不兼容版本。
  "尽力解析"会让旧产物被成功读取但字段全空——表现为"chunk 没有页码"，不报错。
- **偏移一致性是最重要的断言**：`text[char_start:char_end] == content`。
  算错的表现是答案对、高亮错，端到端测试几乎发现不了，只能靠 `verify_chunks()` 入库前拦截。
- **合并块时只从源文本切片**（`parsed.text[start:end]`），**绝不 `"\\n\\n".join(block.text)`**：
  块间分隔符由解析器决定，拼接结果与源文本区间不保证一致。
- **块间只隔空白时可以合并**：不要要求 `left.char_end == right.char_start`（严格连续）。
  解析器插了 `\n\n`，严格连续会让短标题块永远孤立。判据是 `gap.strip()` 为空。
- **短块必须合并**：8 字符的孤立标题块向量几乎无信息量，却占一个检索结果位。
  需要**两遍**合并（先向后、再向前）：首块无前驱、前块是受保护类型时，只能向后并。
- **受保护类型不切分**（`table`/`code`/`formula`）：表格切开行列错位，代码切开失去语义。
  它们也不能并进正文（chunk 类型无法表达），代价是可能产生超长块——这是必须接受的。
- **`fixed` 策略故意不给 bbox**：滑窗边界是任意字符位置，没有对应版面框。
  冒用起点块的 bbox 会让前端框错区域。用 `chunk_metadata.positioning = "char_offset"` 声明。
- **父块不参与检索**（`chunk_level > 0`）：不向量化、检索时按 `level=0` 过滤。
  **A11 的 `search_vector` 必须补上这个过滤**，否则父块与子块同时出现，同一内容占两个结果位。
- **覆盖统计只算叶子块**：父块是子块超集，计入会让重叠率恒 100%、覆盖率恒 100%，
  统计完全失去参考价值。分母也只算非空白字符（块间分隔符不属于任何块是正常的）。
- **语义策略降级必须写日志**：缺 `embed_fn` / 单元太少 / embedding 失败三种情况都退化为
  recursive。不记日志的话，`strategy=semantic` 的文档实际全在跑递归，对比实验数据全是假的。
- **切分参数来自版本快照**（`document_versions.chunking_config`），不是知识库当前默认值。
  用当前配置会让"重跑"与"首次"结果不同，A10 的版本回滚就失去意义。
- **切分落库必须幂等**：同一事务内**先删该版本旧 chunk 再插入**。
  重派任务直接 `bulk_create` 会撞 `uq_chunks_version_index`，
  报错是"唯一约束冲突"，根因却是没做幂等。
- **父子关系分两阶段落库**：chunker 只产出 `parent_index`（批次下标，它不该知道数据库 ID），
  handler 先插全部块拿到 UUID，再回填 `parent_chunk_id`。
- **向量化不进切分步骤**：`embedding` 留 NULL。一起做会让"向量服务临时故障"
  导致切分成果全部丢失；分开后可按 chunk 续跑（`list_pending_embedding`）。

### 4.13 索引层规范（向量存储与零停机重建）

- **向量必须放旁表**（`chunk_embeddings`），不能只放 `chunks.embedding` 单列。
  一行只能存一个向量，"新旧两套向量并存"物理上做不到。
  实测过的错误设计：回填时就地覆盖向量并改 `chunks.index_version_id`
  → 该 chunk 立刻从旧版本查询里消失 → 回填完旧版本就空了
  → 既有停机窗口又无法回滚。详见 docs/adr/0002-index-layer.md。
- **HNSW 索引绝不能用部分索引**（`postgresql_where=...`）。
  `embedding` 列本身 NOT NULL 也不行——"部分索引"这个形式本身
  会让 HNSW 无法服务 `ORDER BY 向量`。实测：完整索引 4.2ms，
  部分索引 131ms 且 `enable_seqscan=off` 也强制不了。**不报错，只是慢 30 倍**。
- **`random_page_cost` 必须按存储介质设**：默认 4.0 是机械硬盘估算，
  会让 planner 放弃 HNSW（191ms vs 7.6ms）。SSD 设 1.1，
  已通过 `DatabaseSettings.random_page_cost` 注入每个连接。
- **检索必须两段式**：先只在旁表上走 HNSW 取候选 ID，再按 ID 取内容并过滤。
  一条 JOIN 会让 `ORDER BY` 距离项来自右表，planner 无法下推索引
  （实测 282ms vs 2.0ms，慢 140 倍）。
  `candidate_multiplier`（默认 4）用于补偿"第一段看不到 chunk 级过滤条件"。
- **`chunk_embeddings` 的唯一约束是回填幂等的保证**：
  `(chunk_id, index_version_id)` 唯一，回填用 `ON CONFLICT DO UPDATE`。
  纯 INSERT 重派会撞约束，报错是"duplicate key"但根因是没做幂等。
- **切换索引版本前必须硬校验回填完成度**（`chunk_count >= expected`）。
  未完成就切换会让检索**静默丢失内容**——比切换失败严重得多。
- **回填完成度以旁表 COUNT 为准**，不是计数器（计数器会滞后或重复累加）。
- **旧版本向量在切换时不能删**：回滚靠它。回收要延迟观察。
- **召回率不能用随机向量测**：1024 维随机单位向量的 top-10 与 top-40
  距离差仅 0.017，边界无区分度，任何近似索引都复现不了。
  `bench_hnsw.py` 只用于验证"索引被 planner 选中"与单查询延迟，
  召回率必须用真实文档 + 黄金集（A15）。

### 4.14 检索层规范（`packages/adapters/src/rag_adapters/retrieval/`）

- **不能靠检索分数拒答**：实测"有答案的最大 top1 距离 0.527"与
  "无答案的最小距离 0.437"区间重叠，任何阈值都会误判
  （0.45 误杀"京沪广深算几线城市"，0.55 放过"食堂几点开饭"）。
  embedding 相似度衡量"语义相近"，不衡量"能否回答"——概念不同。
- **拒答与防幻觉是两个判断，不能混用**：
  - `answerability`（生成前）：材料能否回答这个问题 → **决定拒答**
  - `support`（生成后）：答案中的事实是否都有材料依据 → **拦幻觉**

  混用的实际后果：模型碰巧生成"无法回答"→"有支撑"检查通过
  → 系统记为"正常回答"，误报率统计恒为 100%。
- **每个检索节点一个独立开关**（`RetrievalConfig`），且**构造时校验**：
  开了开关却没给实现必须报错，不能静默降级——静默降级会让评估
  结果"看起来开了但没跑"，是最误导人的情况。
- **RRF 只用排名不用分数**：两路分数量纲不同（余弦距离 0~2 vs
  `ts_rank` 依赖文档长度），加权需归一化，而归一化依赖当前结果集
  分布——同一 chunk 在不同查询里归一化后完全不同，权重含义不可解释。
- **评估必须同批查询、只改一个变量**：多个节点一起上会让指标变化
  无法归因到具体节点。
- **指标满分先怀疑语料**：37-chunk 语料上所有策略都 100% 召回，
  无法区分优劣（metric ceiling）。要比较策略，语料需有
  "近似但错误"的干扰项，留给 A15 用真实业务文档建。
- **提示词的材料必须转义边界标签**：`</document>` → `＜/document＞`，
  否则攻击者写在文档里就能提前闭合标签、把后续内容变成"系统提示"。
- **图用顺序调用而非 `ainvoke`**：线性链路（有开关无条件跳转）下
  顺序调用更易读易测。等真需要"自省不通过→重新检索"的循环时
  再改 `StateGraph`。

### 4.15 引用对齐规范（契约 + 定位）

- **引用必须是契约，不是各端自己拼**：前端要页码/高亮、管理端要审计、
  后端要越权校验，三者需求不同但形状必须一致。字段名漂移
  （`page` vs `page_no`）只会在联调时暴露。契约在
  `rag_contracts.api.citations`，已进 `EXPORTED_MODELS`。
- **`page_no` 从 1 开始**，不用 `page`（含糊）也不用 `page_index`（暗示 0-based）。
- **必须带 `quote` 原文片段**：只给坐标时，解析器没给 bbox（DOCX/XLSX）
  或页面旋转会让前端完全无法高亮。`quote` 让前端能"坐标优先、失败则全文搜"，
  也是人工审计"引用是否错配"的锚点。
- **`relevance_score` 保持距离语义（0 最相关）**，不在契约层转相似度：
  两者语义相反，转换点越多越容易把排序搞反。
- **`citations.index` 必须从 1 连续、不跳号、不重复**：跳号会让前端
  按 index 找引用时拿到错条目，且不报错。
- **定位失败时 `page_no` 必须是 None，不能拿 chunk 起始页冒充**。
  实测：4 页 PDF 里引用第 3 页内容但关联到第 1 页 chunk 时，
  报 `page=1` 会把用户送到完全错误的位置——比"无法跳转"更糟。
  宁可说"不知道在哪一页"。
- **定位判据用"稀有 token 命中"，不是"覆盖率"**：
  - 答案必然含原文没有的词（改写），覆盖率会被拉低（假失败）；
  - 短 chunk 里所有词都"稀有"，必须靠**停用词表**（中英文各一份）
    排除 `must`/`be`/`the`/`城市的"等通用词，否则只共享停用词也会判成功。
- **窗口大小必须自适应 chunk 长度**：固定 200 字符在 60 字的 chunk 上
  退化为"整块"，定位实际没发生（页码永远取起始页）。
- **跨页 chunk 的页码要用"偏移 → 页区间"映射**，不能用 `chunk.page_no`：
  它只记起始页，引用后半段时会翻错页。
- **引用越界（模型引用不存在的编号）丢弃该引用但保留答案**：
  抛异常会让编号错就丢整个答案；静默保留会让前端拿不存在的 index。
  记录 `out_of_range` 计数供管理端观察。
- **引用越权检查是安全兜底**：SQL 已过滤，但模型可能记错来源、
  压缩/回填可能引入未过滤内容。引用越界是最隐蔽的泄漏形态——
  它嵌在答案里。`assemble_citations` 的 `visible_document_ids` 参数必传。
