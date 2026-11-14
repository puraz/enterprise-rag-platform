"""Redis 连接配置。

本机跑的是 Valkey（Redis 协议兼容），所以这里仍按 Redis 语义写。
"""

from __future__ import annotations

from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class RedisSettings(BaseModel):
    """Redis / Valkey 连接参数。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    host: str = "127.0.0.1"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    # 密码可选：本地不开鉴权就是 None，生产必须设
    password: SecretStr | None = None

    socket_timeout_s: float = Field(default=3.0, gt=0)
    # 信号量兜底：并发放大时先在这里限住，别把 Redis 连接数打满
    max_connections: int = Field(default=50, ge=1)

    @property
    def url(self) -> SecretStr:
        """redis:// 连接串（含密码），同样是 SecretStr。"""
        auth = ""
        if self.password is not None:
            auth = f":{quote(self.password.get_secret_value(), safe='')}@"
        return SecretStr(f"redis://{auth}{self.host}:{self.port}/{self.db}")

    @property
    def redacted_url(self) -> str:
        """给日志和诊断用的脱敏连接串。"""
        auth = ":***@" if self.password is not None else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
