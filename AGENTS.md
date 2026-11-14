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

后端分层（apps/api 内）：api(路由) / schemas / services(业务) / deps /
          pipelines / retrievers / parsers / chunkers / storages /
          providers / identity / evaluators / workers
前端分层：routes / features / components / hooks / api(自动生成，别手改) / stores
- 路由里不写业务逻辑；service 里不写 SQL；SQL 只在 packages/db 的 repository。
- 依赖方向单向向内，禁止反向 import。

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

## 5. 禁区（never do）
- 不手改 packages/contracts 里自动生成的前端类型（改源头 schema 再 gen）。
- 不把 API Key、真实租户数据、用户文档内容写进日志/异常/trace/测试 fixture。
- 不新增同步阻塞调用到事件循环里（含 boto3 直调，要放线程池）。
- 不用 OFFSET 分页（用 keyset）；不写 `SELECT *`；不在循环里查库(N+1)。
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
