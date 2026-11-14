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
    user: str = "rag"
    # 开发默认值故意写成一眼能认出来的样子，便于生产守卫拦截
    password: SecretStr = SecretStr("rag_local_dev_only")
    name: str = "rag"

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
    def redacted_dsn(self) -> str:
        """给日志和诊断用的脱敏 DSN。"""
        user = quote(self.user, safe="")
        return f"postgresql+asyncpg://{user}:***@{self.host}:{self.port}/{self.name}"
