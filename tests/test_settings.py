"""配置体系测试：优先级、嵌套映射、密钥脱敏、生产守卫、.env 加载。

每个用例都在防一类具体事故，见各 docstring。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_core.settings import RunEnv, Settings, get_settings, load_env_files


@pytest.fixture(autouse=True)
def _isolate_env() -> Iterator[None]:
    """彻底隔离环境：清空 RAG_* 并在测试后整体还原。

    不隔离的话，开发者 shell 里导出的 RAG_* 会让测试时红时绿，
    这类"只在某些机器上失败"的测试最消耗信任。
    """
    snapshot = dict(os.environ)
    for key in list(os.environ):
        if key.startswith("RAG_"):
            del os.environ[key]
    get_settings.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(snapshot)
    get_settings.cache_clear()


def test_code_defaults_are_used_when_nothing_is_set() -> None:
    """没有任何环境变量时，用代码默认值（本地可跑）。"""
    settings = Settings()
    assert settings.env is RunEnv.LOCAL
    assert settings.database.port == 5432
    assert settings.resolved_log_format == "console"


def test_process_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """进程环境变量必须能覆盖代码默认值，否则平台注入配置失效。"""
    monkeypatch.setenv("RAG_DATABASE__HOST", "db.prod.internal")
    monkeypatch.setenv("RAG_DATABASE__PORT", "6543")
    settings = Settings()
    assert settings.database.host == "db.prod.internal"
    assert settings.database.port == 6543


def test_unprefixed_variable_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有 RAG_ 前缀的同名变量不能被误读，避免不同系统间变量串味。"""
    monkeypatch.setenv("DATABASE__HOST", "should-not-be-read")
    assert Settings().database.host == "127.0.0.1"


def test_unknown_nested_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """拼错的配置项必须报错，而不是静默回退默认值。"""
    monkeypatch.setenv("RAG_DATABASE__HOSS", "typo")
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_never_appear_in_safe_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置快照可以上报/打印，但绝不能带出密钥。"""
    monkeypatch.setenv("RAG_DATABASE__PASSWORD", "s3cr3t-value-xyz")
    monkeypatch.setenv("RAG_REDIS__PASSWORD", "redis-secret-xyz")
    dumped = json.dumps(Settings().safe_dump(), ensure_ascii=False)
    assert "s3cr3t-value-xyz" not in dumped
    assert "redis-secret-xyz" not in dumped
    assert "**********" in dumped


def test_dsn_url_encodes_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """密码里有 @ : / 时必须编码，否则 DSN 会被解析成别的 host。"""
    monkeypatch.setenv("RAG_DATABASE__PASSWORD", "p@ss/w:rd")
    dsn = Settings().database.dsn.get_secret_value()
    assert "%40" in dsn  # @
    assert "%2F" in dsn  # /


def test_production_rejects_development_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """生产环境带开发默认密码必须启动失败——这是复制模板最容易带上去的雷。"""
    monkeypatch.setenv("RAG_ENV", "prod")
    monkeypatch.setenv("RAG_IDENTITY__JWT_SECRET", "a-production-grade-secret-value-32+")
    monkeypatch.setenv("RAG_IDENTITY__JWT_ALGORITHM", "RS256")
    with pytest.raises(ValidationError, match="生产环境配置不安全"):
        Settings()


def test_production_rejects_debug_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """生产开 debug 会泄漏调用栈与内部路径，必须拦。"""
    monkeypatch.setenv("RAG_ENV", "prod")
    monkeypatch.setenv("RAG_IDENTITY__JWT_SECRET", "a-production-grade-secret-value-32+")
    monkeypatch.setenv("RAG_IDENTITY__JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("RAG_DATABASE__PASSWORD", "a-real-strong-secret")
    monkeypatch.setenv("RAG_DEBUG", "true")
    with pytest.raises(ValidationError, match="RAG_DEBUG"):
        Settings()


def test_production_with_real_secret_passes_and_uses_json_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产配置合法时应当通过，且日志自动切 JSON。"""
    monkeypatch.setenv("RAG_ENV", "prod")
    monkeypatch.setenv("RAG_IDENTITY__JWT_SECRET", "a-production-grade-secret-value-32+")
    monkeypatch.setenv("RAG_IDENTITY__JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("RAG_DATABASE__PASSWORD", "a-real-strong-secret")
    settings = Settings()
    assert settings.env is RunEnv.PROD
    assert settings.resolved_log_format == "json"


def test_load_env_files_layering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """.env 提供基线，.env.<env> 覆盖同名项，且都能被 Settings 读到。"""
    (tmp_path / ".env").write_text(
        "RAG_DATABASE__HOST=from-base\nRAG_DATABASE__PORT=1111\n", encoding="utf-8"
    )
    (tmp_path / ".env.dev").write_text("RAG_DATABASE__PORT=2222\n", encoding="utf-8")
    monkeypatch.setenv("RAG_ENV", "dev")

    loaded = load_env_files(tmp_path)

    assert {path.name for path in loaded} == {".env", ".env.dev"}
    settings = Settings()
    assert settings.database.host == "from-base"
    assert settings.database.port == 2222  # 覆盖文件赢


def test_load_env_files_does_not_override_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.env 只是开发便利，绝不能打赢真实环境变量。"""
    (tmp_path / ".env").write_text("RAG_DATABASE__HOST=from-file\n", encoding="utf-8")
    monkeypatch.setenv("RAG_DATABASE__HOST", "from-real-env")

    load_env_files(tmp_path)

    assert os.environ["RAG_DATABASE__HOST"] == "from-real-env"


def test_load_env_files_reads_selector_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAG_ENV 写在 .env 里也要能正确选中 .env.<env>（两趟读）。"""
    (tmp_path / ".env").write_text("RAG_ENV=dev\n", encoding="utf-8")
    (tmp_path / ".env.dev").write_text("RAG_DATABASE__PORT=3333\n", encoding="utf-8")

    load_env_files(tmp_path)

    assert Settings().database.port == 3333
    assert Settings().env is RunEnv.DEV
