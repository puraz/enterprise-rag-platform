"""配置根：全后端唯一的配置入口。

优先级（自上而下递减）：
    进程环境变量 > .env.<RAG_ENV> > .env > 代码默认值

命名规则：前缀 RAG_，嵌套用双下划线，例如
    RAG_DATABASE__HOST  ->  settings.database.host
    RAG_REDIS__PORT     ->  settings.redis.port
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_core.settings.database import DatabaseSettings
from rag_core.settings.redis import RedisSettings

# RAG_ENV 要参与"选哪个 .env 文件"，所以常量单独暴露
ENV_SELECTOR = "RAG_ENV"

# 出现在模板里、必须视为"绝不允许进生产"的已知开发密钥
DEV_SECRET_VALUES = frozenset({"rag_local_dev_only", "changeme", "password"})

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class RunEnv(StrEnum):
    """运行环境。决定 .env 覆盖文件、日志格式、生产守卫是否生效。"""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

    @property
    def is_production_like(self) -> bool:
        """staging 也按生产标准校验：预发就是用来提前暴露这类问题的。"""
        return self in (RunEnv.STAGING, RunEnv.PROD)


class Settings(BaseSettings):
    """全量配置。新增能力域时在这里挂一个子配置，不要散装 os.getenv。"""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        # RAG_DATABASE__HOST 里的 __ 就是这行定义的
        env_nested_delimiter="__",
        # 故意不用内置 env_file：文件加载由 load_env_files() 显式完成
        env_file=None,
        case_sensitive=False,
        # 未知的 RAG_* 变量直接报错，挡住拼写错误
        extra="forbid",
        validate_default=True,
    )

    env: RunEnv = RunEnv.LOCAL
    app_name: str = "enterprise-rag-platform"
    debug: bool = False
    log_level: LogLevel = "INFO"
    # 留空表示"按环境推导"，见 resolved_log_format
    log_format: Literal["json", "console"] | None = None

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    @property
    def resolved_log_format(self) -> Literal["json", "console"]:
        """日志格式推导：生产必须 JSON（可被采集器解析），本地给人看。"""
        if self.log_format is not None:
            return self.log_format
        return "json" if self.env.is_production_like else "console"

    @model_validator(mode="after")
    def _guard_production_safety(self) -> Self:
        """生产环境守卫：宁可启动失败，也不要带着开发配置上线。"""
        if not self.env.is_production_like:
            return self

        problems: list[str] = []

        db_password = self.database.password.get_secret_value()
        if db_password in DEV_SECRET_VALUES:
            problems.append("RAG_DATABASE__PASSWORD 仍是开发默认值")

        if self.redis.password is not None:
            redis_password = self.redis.password.get_secret_value()
            if redis_password in DEV_SECRET_VALUES:
                problems.append("RAG_REDIS__PASSWORD 仍是开发默认值")

        if self.debug:
            problems.append("RAG_DEBUG=true 不允许出现在生产")

        if problems:
            joined = "；".join(problems)
            raise ValueError(f"生产环境配置不安全：{joined}")

        return self

    def safe_dump(self) -> dict[str, Any]:
        """导出可安全打印/上报的配置快照。

        model_dump(mode="json") 会把 SecretStr 序列化成 **********，
        再补上脱敏 DSN，方便排查"这个 pod 到底连的哪个库"。
        """
        data = self.model_dump(mode="json")
        data["database"]["redacted_dsn"] = self.database.redacted_dsn
        data["redis"]["redacted_url"] = self.redis.redacted_url
        return data


def load_env_files(project_root: Path | None = None) -> tuple[Path, ...]:
    """把 .env / .env.<RAG_ENV> 合并后注入进程环境。

    两个关键决策：
    1. 用 os.environ.setdefault 注入 —— 真实环境变量永远赢，.env 只是开发便利；
    2. RAG_ENV 两趟读 —— 先看进程环境，再看 .env，否则无法确定加载哪个覆盖文件。

    返回实际加载的文件列表，便于启动日志打印"本次生效的配置文件"。
    """
    root = project_root if project_root is not None else Path.cwd()

    base_file = root / ".env"
    base_values: dict[str, str] = {}
    if base_file.is_file():
        base_values = {
            key: value for key, value in dotenv_values(base_file).items() if value is not None
        }

    # 第一趟已经读到 .env；这里才决定覆盖文件名
    env_name = os.getenv(ENV_SELECTOR) or base_values.get(ENV_SELECTOR) or RunEnv.LOCAL.value

    merged: dict[str, str] = dict(base_values)
    loaded: list[Path] = [base_file] if base_file.is_file() else []

    override_file = root / f".env.{env_name}"
    if override_file.is_file():
        for key, value in dotenv_values(override_file).items():
            if value is not None:
                merged[key] = value  # 后读的文件覆盖先读的
        loaded.append(override_file)

    # 把环境名钉死：防止 .env.<env> 里再写 RAG_ENV 造成"文件选的是 dev、实际值却是 prod"
    merged[ENV_SELECTOR] = env_name

    for key, value in merged.items():
        os.environ.setdefault(key, value)

    return tuple(loaded)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例。测试里改完环境变量要 get_settings.cache_clear()。"""
    load_env_files()
    return Settings()


def reload_settings() -> Settings:
    """显式刷新配置（管理端改配置后热加载用）。"""
    get_settings.cache_clear()
    return get_settings()
