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

- **成功响应 = 裸数据**，不做 `{code, message, data}` 外层包裹。
  错误靠 HTTP 状态码 + `ErrorResponse.code` 分支。
- **错误响应唯一形状** = `rag_contracts.errors.ErrorResponse`：
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
