"""PostgreSQL 连接配置。

只放"接哪儿 / 怎么连"的参数，不含业务概念。SQL 与表结构归 A3。
"""

from __future__ import annotations

from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class DatabaseSettings(BaseModel):
    """PostgreSQL 连接与连接池参数。"""

    # extra=forbid：配置项写错（RAG_DATABASE__HOSS）立即报错，不静默用默认值
    # frozen=True：运行期禁止改配置，避免"某处代码悄悄改了 DSN"这类事故
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    host: str = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    # 默认用应用角色 rag_app：**必须是非超级用户**，否则 RLS 完全失效。
    # 超级用户（rolsuper）与 BYPASSRLS 角色都会无条件绕过行级安全策略，
    # 且 pg_class.relforcerowsecurity 对 superuser 也无效。
    # 迁移用单独的 rag_admin/puraz（需要 DDL 权限），见 alembic/env.py。
    user: str = "rag_app"
    # 开发默认值故意写成一眼能认出来的样子，便于生产守卫拦截
    password: SecretStr = SecretStr("rag_local_dev_only")
    name: str = "rag"

    # 迁移专用凭据（可选）。不设则复用上面的 user/password。
    # 生产环境应显式设置成高权限角色，与运行时角色分离。
    migration_user: str | None = None
    migration_password: SecretStr | None = None

    # 管理角色（BYPASSRLS）：用于合法的跨租户路径——
    # 首登开通租户/成员、管理端看板、后台对账。
    # 不配则 `Database.transaction_bypass_rls()` 会明确报错。
    admin_user: str | None = None
    admin_password: SecretStr | None = None

    # 连接池：这三个数字直接决定"并发上来时会不会把 DB 打爆"
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout_s: float = Field(default=5.0, gt=0)

    # 超时兜底：没有这两条，一个慢查询能把整条连接链拖死
    statement_timeout_ms: int = Field(default=15_000, ge=0)
    lock_timeout_ms: int = Field(default=5_000, ge=0)

    @property
    def dsn(self) -> SecretStr:
        """SQLAlchemy async 使用的完整 DSN（含密码）。

        返回 SecretStr 而不是 str：即使有人把它塞进日志/异常/trace，
        也只会看到 **********。要用真值必须显式 get_secret_value()。
        """
        user = quote(self.user, safe="")
        # 密码必须 URL 编码：@ / : / 出现在密码里不编码会把 DSN 解析坏
        password = quote(self.password.get_secret_value(), safe="")
        raw = f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/{self.name}"
        return SecretStr(raw)

    @property
    def migration_dsn(self) -> SecretStr:
        """迁移专用 DSN（高权限角色）。

        **为什么迁移要单独的角色**：
        运行时角色 `rag_app` 只有 DML 权限（SELECT/INSERT/UPDATE/DELETE），
        没有 CREATE TABLE / ALTER TABLE / CREATE POLICY 权限 —— 这是有意的
        最小权限设计。迁移需要 DDL，必须用另一个角色。

        用环境变量 `RAG_DATABASE__MIGRATION_USER/PASSWORD` 覆盖；
        未设置时回退到主 user/password（方便单角色环境）。

        **不要用超级用户跑应用**：超级用户无条件绕过 RLS，
        策略会静默失效。迁移用高权限、运行时用低权限是正确的分工。
        """
        user = quote(self.migration_user or self.user, safe="")
        password_value = (
            self.migration_password.get_secret_value()
            if self.migration_password is not None
            else self.password.get_secret_value()
        )
        password = quote(password_value, safe="")
        raw = f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/{self.name}"
        return SecretStr(raw)

    @property
    def admin_dsn(self) -> SecretStr | None:
        """管理角色 DSN（BYPASSRLS）。未配置时返回 None。

        **绝不能把它当默认连接串用**：绕过 RLS 意味着任何漏写
        `WHERE tenant_id` 的查询都会读到全部租户的数据。
        """
        if not self.admin_user:
            return None
        user = quote(self.admin_user, safe="")
        password_value = (
            self.admin_password.get_secret_value()
            if self.admin_password is not None
            else self.password.get_secret_value()
        )
        password = quote(password_value, safe="")
        raw = f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/{self.name}"
        return SecretStr(raw)

    @property
    def redacted_dsn(self) -> str:
        """给日志和诊断用的脱敏 DSN。"""
        user = quote(self.user, safe="")
        return f"postgresql+asyncpg://{user}:***@{self.host}:{self.port}/{self.name}"
