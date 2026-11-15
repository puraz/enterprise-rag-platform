"""RLS（行级安全）策略的 SQL 定义。

## 为什么 RLS 是"最后一道防线"而不是唯一防线

应用层必须自己写 `WHERE tenant_id = :t`（仓储层已经这么做），
RLS 的作用是"万一某处漏写了，数据库还能兜住"。

两者都要有：
- 只有应用层过滤 → 一处遗漏就是越权事故；
- 只有 RLS → 开发时看不出问题，但每次查询多一层开销，
  且某些场景（管理员跨租户、后台聚合）必须绕过，策略会变复杂。

## 会话变量的注入方式（本步最关键的一点）

RLS 策略依赖 `current_setting('app.tenant_id', true)`。
注入它有正确与错误两种做法：

**错误做法**（会造成真实越权）：
    SET app.tenant_id = '<uuid>'        -- 会话级，连接归还池后仍然生效
    ...
    -- 下一个请求复用了这条连接，如果它因为异常没设置 tenant_id，
    -- current_setting 会返回上一个租户的 ID → 读到别人的数据

**正确做法**：
    BEGIN;
    SET LOCAL app.tenant_id = '<uuid>';  -- 只在本事务内有效
    ... 查询 ...
    COMMIT;                              -- 事务结束自动清除

或者用 `set_config('app.tenant_id', :v, true)`（第三个参数 `true` = is_local）。

**还有一种更隐蔽的错误**：在连接池的 `after_begin` 钩子里设 `SET`（非 LOCAL），
然后以为"每个请求都会重设所以安全"。这在异步/并发下不成立——
connection pool 的 checkout/checkin 顺序不保证，且异常路径可能跳过设置。

本项目统一用 `SET LOCAL`，且**必须在事务内**执行（`SET LOCAL` 在事务外执行
会报 warning 且立即失效）。

## 未设置 tenant_id 时的行为（实测修正）

**重要实测发现**：`current_setting('app.tenant_id', true)` 在变量未设置时
返回**空字符串 `''`**，不是 NULL。直接 `''::uuid` 会报错：

    invalid input syntax for type uuid: ""

这个错误会以 500 的形式冒到用户面前，而不是"查不到数据"。

因此策略必须显式处理空字符串：

    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)

`NULLIF(x, '')` 把空字符串转成 NULL，`tenant_id = NULL` 结果为 NULL（非真），
从而**拒绝所有行** —— 这才是安全的默认。

这个坑很隐蔽：如果只测"设了 tenant 时的隔离"（会通过），
而未测"没设 tenant 时的行为"，上线后任何漏设上下文的路由都会 500。
"""

from __future__ import annotations

# 所有需要 RLS 的表。新增租户内表时必须加进这里。
#
# 不在列表里的表：
# - tenants：租户表本身，按 id 而非 tenant_id 过滤，单独处理
# - users / identity_links / audit_chain_heads（tenant_id 可空或全局）
# - schema_health / alembic_version：元数据表
RLS_TABLES: tuple[str, ...] = (
    "knowledge_bases",
    "documents",
    "document_versions",
    "blobs",
    "chunks",
    "index_versions",
    "upload_sessions",
    "tenant_memberships",
    "tasks",
    "outbox_messages",
    "model_calls",
)

# 审计日志单独处理：tenant_id 可为 NULL（全局事件），
# 策略要允许平台超管看到 NULL 行。
AUDIT_TABLE = "audit_logs"

# 会话变量名。统一前缀 `app.` 避免与 PG 内建参数冲突。
TENANT_SETTING = "app.tenant_id"


def enable_rls_statements(table: str) -> list[str]:
    """生成启用 RLS 的语句。

    `ENABLE ROW LEVEL SECURITY` 让策略对**普通角色**生效；
    表所有者（puraz）默认仍绕过，除非再执行 `FORCE ROW LEVEL SECURITY`。
    但 **superuser 始终绕过，FORCE 也不管用** —— 所以应用必须用
    非超级用户（rag_app）连库，否则 RLS 完全无效。
    """
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        # FORCE 让表所有者（若应用用所有者连库）也受约束。
        # 对 superuser 无效，但保留它能在"应用改用所有者角色"时提供保护。
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
    ]


def current_tenant_sql() -> str:
    """生成"从会话变量取当前租户"的 SQL 片段。

    **必须用 NULLIF 处理空字符串**：未设置时 `current_setting(..., true)`
    返回 `''` 而不是 NULL，直接 `::uuid` 会报
    `invalid input syntax for type uuid: ""`（实测确认）。

    用 NULLIF 转成 NULL 后，`tenant_id = NULL` 为 NULL（非真），
    自然拒绝所有行 —— 安全的默认行为。
    """
    return f"NULLIF(current_setting('{TENANT_SETTING}', true), '')::uuid"


def create_tenant_isolation_policy_statements(table: str) -> list[str]:
    """生成租户隔离策略。

    ## 关于 USING 与 WITH CHECK（实测澄清）

    两者语义不同：
    - `USING`：哪些**已存在的行**对 SELECT / UPDATE / DELETE 可见；
    - `WITH CHECK`：INSERT 或 UPDATE 后，**新行**必须满足什么。

    **实测发现**（PG 18）：当策略只写 `USING` 时，PG 会用同一个表达式
    检查新行 —— 跨租户 INSERT 与 UPDATE 都会被拒（报"新行违背了行级安全策略"）。
    所以本项目的租户隔离场景里，只写 `USING` 也能防住越权写。

    **那为什么还写 `WITH CHECK`**：
    1. **表达意图**：读取隔离和写入校验是两回事，显式写出便于审计与 review；
    2. **未来分化的保护**：一旦有人把 `USING` 放宽（比如"软删除的行也可见"），
       没有 `WITH CHECK` 会连带放宽写入校验，形成漏洞窗口；
    3. 部分场景（如允许读全部但只能写自己的）必须靠两者不同来表达。

    测试 `test_rls.py` 覆盖了 INSERT 与 UPDATE 两条越权写路径。
    """
    tenant_expr = current_tenant_sql()
    return [
        f"DROP POLICY IF EXISTS tenant_isolation ON {table}",
        f"""
        CREATE POLICY tenant_isolation ON {table}
        USING (tenant_id = {tenant_expr})
        WITH CHECK (tenant_id = {tenant_expr})
        """.strip(),
    ]


def create_bypass_statements(table: str, bypass_role: str = "rag_admin") -> list[str]:
    """给管理角色创建绕过策略。

    有些合法场景需要跨租户：
    - 管理端看板（统计所有租户）；
    - 后台对账任务；
    - 数据迁移。

    做法：创建一个带 `BYPASSRLS` 的角色（`rag_admin`），
    应用只在明确需要跨租户的路径上用它连库，其它路径用 `rag_app`。

    **不要用"策略里判断角色名"的方式**：
        USING (current_user = 'rag_admin' OR tenant_id = ...)
    这种写法让策略依赖当前角色名，一旦有人用错连接串，
    越权与不越权完全取决于部署配置，极难审计。
    """
    return [
        f"DROP POLICY IF EXISTS {table}_bypass ON {table}",
        f"""
        CREATE POLICY {table}_bypass ON {table}
        TO {bypass_role}
        USING (true)
        WITH CHECK (true)
        """.strip(),
    ]


def audit_policy_statements(bypass_role: str = "rag_admin") -> list[str]:
    """审计日志的 RLS 策略（特殊：tenant_id 可为 NULL）。

    规则：
    - 普通角色：只能看自己租户的记录；
    - 管理角色：能看全部（含 tenant_id IS NULL 的全局事件）。

    为什么普通角色看不到 NULL 行：登录失败这类事件可能包含
    攻击者尝试的租户信息，不该让任何租户看到。
    """
    tenant_expr = current_tenant_sql()
    return [
        f"DROP POLICY IF EXISTS tenant_isolation ON {AUDIT_TABLE}",
        f"DROP POLICY IF EXISTS {AUDIT_TABLE}_bypass ON {AUDIT_TABLE}",
        f"""
        CREATE POLICY tenant_isolation ON {AUDIT_TABLE}
        USING (tenant_id = {tenant_expr})
        WITH CHECK (tenant_id = {tenant_expr})
        """.strip(),
        f"""
        CREATE POLICY {AUDIT_TABLE}_bypass ON {AUDIT_TABLE}
        TO {bypass_role}
        USING (true)
        WITH CHECK (true)
        """.strip(),
    ]
