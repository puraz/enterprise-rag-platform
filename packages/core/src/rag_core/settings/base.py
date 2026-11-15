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
from rag_core.settings.embedding import EmbeddingSettings
from rag_core.settings.identity import IdentitySettings, OIDCSettings
from rag_core.settings.llm import LLMSettings
from rag_core.settings.redis import RedisSettings
from rag_core.settings.storage import StorageSettings
from rag_core.settings.worker import WorkerSettings

# RAG_ENV 要参与"选哪个 .env 文件"，所以常量单独暴露
ENV_SELECTOR = "RAG_ENV"

# 出现在模板里、必须视为"绝不允许进生产"的已知开发密钥。
# 123456 这类弱口令同理：本地图省事可以，生产必须拦。
DEV_SECRET_VALUES = frozenset({"rag_local_dev_only", "changeme", "password", "123456", "root"})

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

    # ---- 开发用身份提供者 ----
    # 默认关闭：生产环境绝不该有这个能力。
    # 开启后 `/api/v1/auth/dev-login` 才会被注册，且接受 `dev:<subject>` 凭证。
    # 三重防护见 DevIdentityProvider 的 docstring。
    dev_auth_enabled: bool = False

    # 启动时是否等待并校验依赖（PG/Redis）。
    # 自动化测试（尤其只测 HTTP 契约的用例）可以关掉它，
    # 让测试不依赖真实数据库，更快也更稳定。
    # **生产绝不该关**：那样进程会在依赖不可用时"启动成功"，
    # 所有请求都失败，而编排器以为实例健康。
    startup_dependency_check: bool = True

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    oidc: OIDCSettings = Field(default_factory=OIDCSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)

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

        # 开发身份提供者绝不能在生产开启：
        # 它接受 `dev:<任意subject>` 就能变成任意用户，等于无鉴权。
        if self.dev_auth_enabled:
            problems.append("RAG_DEV_AUTH_ENABLED=true 不允许出现在生产")

        # JWT 签名密钥必须换掉默认值，否则任何人都能伪造 token
        if self.identity.jwt_secret in DEV_SECRET_VALUES or len(self.identity.jwt_secret) < 32:
            problems.append("RAG_IDENTITY__JWT_SECRET 仍是默认值或长度不足 32")

        # 生产必须用非对称签名：HS256 要求所有验证方共享密钥，
        # 任何一个服务被攻破就等于签发权泄漏
        if self.identity.jwt_algorithm == "HS256":
            problems.append("生产环境应使用 RS256 等非对称算法，而非 HS256")

        # 生产必须做启动依赖检查：
        # 关掉它意味着依赖挂掉时进程"启动成功"，编排器以为实例健康，
        # 但所有请求都 500 —— 比启动失败难排查得多。
        if not self.startup_dependency_check:
            problems.append("RAG_STARTUP_DEPENDENCY_CHECK=false 不允许出现在生产")

        # LLM 供应商密钥：配了 provider 就必须有 key，否则所有 AI 功能不可用
        if self.llm.provider and not self.llm.api_key:
            problems.append(f"RAG_LLM__API_KEY 未配置（provider={self.llm.provider}）")

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


def find_project_root(start: Path | None = None) -> Path:
    """向上查找仓库根目录。

    为什么必须这么做：`.env` 在仓库根，但命令可能从任意子目录执行
    （`packages/db` 下跑 alembic、`apps/api` 下跑 uvicorn、IDE 里跑测试）。
    只看 `Path.cwd()` 会导致找不到 `.env`，然后**静默退回代码默认值**
    ——表现是"连到了错误的库/用了默认密码"，报错信息完全指不到根因。

    生产环境（容器/sytemd）通常没有仓库结构：找不到标记目录时
    原样返回起始目录，此时只靠进程环境变量，这是正确的行为——
    生产配置必须由平台注入，不该依赖文件。

    标记：同时有 pyproject.toml 与 uv.lock（uv workspace 的两个特征文件）。
    只认 pyproject.toml 会在子包里误判（每个子包都有自己的 pyproject.toml）。
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "uv.lock").is_file():
            return candidate
    return current


def load_env_files(project_root: Path | None = None) -> tuple[Path, ...]:
    """把 .env / .env.<RAG_ENV> 合并后注入进程环境。

    两个关键决策：
    1. 用 os.environ.setdefault 注入 —— 真实环境变量永远赢，.env 只是开发便利；
    2. RAG_ENV 两趟读 —— 先看进程环境，再看 .env，否则无法确定加载哪个覆盖文件。

    project_root 不传时自动向上查找仓库根，见 find_project_root 的说明。
    返回实际加载的文件列表，便于启动日志打印"本次生效的配置文件"。
    """
    root = project_root if project_root is not None else find_project_root()

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
