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
- packages/core        `rag_core`        横切基础设施（配置/日志/异常/DB池/Redis/韧性），不含业务
- packages/contracts   `rag_contracts`   Pydantic 契约 + 枚举 + 错误码 + 队列消息（叶子包，零内部依赖）
- packages/db          `rag_db`          ORM 模型 + 仓储(SQL) + Alembic 迁移
- apps/api             `rag_api`         路由/服务/依赖注入
- apps/worker          `rag_worker`      管线/定时任务

**为什么有 packages/db（不能省略，也不能塞进别处）**：
ORM 模型与仓储必须被 api 与 worker **同时**使用（worker 要改任务状态、写 chunk、
读文档元数据），但「worker 禁止 import apps/api」是硬约束，而 core 又规定
「不含任何业务概念」。三条约束叠加后，模型与仓储只能落在第三个共享包里。

依赖方向（严格单向，`tests/test_workspace.py` 会检查）：
```
contracts ← core ← db ← api / worker
```
api 与 worker **互不依赖**，只通过契约、数据库状态、队列消息通信。

后端分层（apps/api 内）：api(路由) / schemas / services(业务) / deps(依赖注入) /
          pipelines / retrievers / parsers / chunkers / storages /
          providers / identity / evaluators / workers / openapi
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
  共享代码只能进 packages/contracts、packages/core 或 packages/db。
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
