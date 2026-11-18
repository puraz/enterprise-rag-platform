"""日志层测试：脱敏覆盖度、上下文透传、跨线程传播、JSON 可解析。

重点在"防真实事故"，不是覆盖率：
每一条 assert 都对应一种在生产里会变成安全事故或排查噩梦的写法。
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from collections.abc import Iterator

import pytest
import structlog

from rag_core.logging import (
    MASK,
    bind_context,
    clear_context,
    contextual_call,
    get_logger,
    new_request_id,
    redact_processor,
    request_context,
    setup_logging,
)
from rag_core.settings import Settings


@pytest.fixture
def json_logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    """把日志输出到内存流，方便直接断言内容。

    **必须隔离环境变量**：任何测试调用 `get_settings()` 都会触发
    `load_env_files()`，把 `.env` 内容注入 `os.environ`。之后构造
    `Settings(env="prod", ...)` 会读到 `.env` 里的 `RAG_DEBUG=true`、
    `RAG_DEV_AUTH_ENABLED=true`，撞上生产守卫。

    这类"单独跑通过、全量跑失败"的测试污染极难排查——
    测试之间必须互不影响。
    """
    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)

    stream = io.StringIO()
    settings = Settings(
        env="prod",
        log_format="json",
        database={"password": "a-real-strong-secret"},
        identity={"jwt_secret": "a-production-grade-secret-value-32+", "jwt_algorithm": "RS256"},
        llm={"api_key": "sk-a-real-production-key-value"},
        # Embedding 现在是远程 API：生产守卫要求有 key
        embedding={"api_key": "a-real-embedding-key"},
    )
    setup_logging(settings, stream=stream)
    yield stream
    clear_context()
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


def _last_json_line(stream: io.StringIO) -> dict[str, object]:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "没有任何日志输出"
    # 只取最后一行，因为前面可能有第三方 logger 的初始化噪音
    return json.loads(lines[-1])


# ------------------------------------------------------------ 脱敏：键名


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "PASSWORD",
        "api_key",
        "X-Api-Key",
        "x_api_key",
        "authorization",
        "Authorization",
        "refresh_token",
        "client_secret",
        "private_key",
        "dsn",
        "redis_url",
        "session_token",
        # 裸 token 是凭据（JWT/API token），必须脱敏
        "token",
    ],
)
def test_sensitive_keys_are_fully_masked(key: str) -> None:
    """任何形式的密钥键名都整体替换，且大小写/连接符变体都要命中。"""
    result = redact_processor(None, "info", {"event": "x", key: "super-secret-value"})  # type: ignore[arg-type]
    assert result[key] == MASK


@pytest.mark.parametrize("key", ["input_tokens", "output_tokens", "total_tokens", "cached_tokens"])
def test_token_usage_counters_survive_redaction(key: str) -> None:
    """计费 token 计数必须保留。

    这是"脱敏不能过度"的边界：一刀切把 *_token 都脱敏掉，
    成本看板和账单对账全部失效，而且极难发现（日志看着"很安全"）。
    """
    result = redact_processor(None, "info", {"event": "llm_call", key: 1234})  # type: ignore[arg-type]
    assert result[key] == 1234


def test_nested_structure_is_redacted() -> None:
    """嵌套 dict / list 里的密钥同样要脱敏，不能只看第一层。"""
    event = {
        "event": "call",
        "request": {
            "headers": {"Authorization": "Bearer abc123", "Accept": "application/json"},
            "body": [{"api_key": "k-1"}, {"ok": "fine"}],
        },
    }
    result = redact_processor(None, "info", event)  # type: ignore[arg-type]
    assert result["request"]["headers"]["Authorization"] == MASK  # type: ignore[index]
    assert result["request"]["headers"]["Accept"] == "application/json"  # type: ignore[index]
    assert result["request"]["body"][0]["api_key"] == MASK  # type: ignore[index]
    assert result["request"]["body"][1]["ok"] == "fine"  # type: ignore[index]


# ------------------------------------------------------- 脱敏：值的模式


def test_password_inside_url_is_masked() -> None:
    """DSN 被拼进字符串时也要脱敏——这是按 key 判断挡不住的最大盲区。

    真实写法：log.info(f"connecting to {settings.database.dsn}")
    此时 key 是 "event"，里面藏着密码。
    """
    result = redact_processor(
        None,
        "info",
        {"event": "connecting postgresql+asyncpg://rag:s3cr3t@db:5432/rag"},
    )  # type: ignore[arg-type]
    assert "s3cr3t" not in str(result["event"])
    assert "rag:***@db" in str(result["event"])


def test_bearer_token_in_message_is_masked() -> None:
    """日志正文里出现 Bearer 凭据必须打码。"""
    result = redact_processor(
        None, "info", {"event": "GET /api user-agent=curl Authorization: Bearer eyJhbGciOi.abc.def"}
    )  # type: ignore[arg-type]
    assert "eyJhbGciOi" not in str(result["event"])
    assert "Bearer ***" in str(result["event"])


def test_self_referencing_dict_does_not_hang() -> None:
    """自引用结构必须有深度上限，否则日志层能把进程挂死。"""
    loop: dict[str, object] = {"event": "x"}
    loop["self"] = loop
    result = redact_processor(None, "info", loop)  # type: ignore[arg-type]
    assert "<max-depth>" in json.dumps(result)


# ------------------------------------------------------- contextvars 透传


def test_context_fields_appear_in_every_log_line(json_logs: io.StringIO) -> None:
    """request_id/tenant_id/user_id 自动出现在该上下文内所有日志里。"""
    log = get_logger("test")
    with request_context(tenant_id="t-1", user_id="u-1") as request_id:
        log.info("first")
        log.info("second", doc_id="d-1")

    lines = [json.loads(raw) for raw in json_logs.getvalue().splitlines() if raw.strip()]
    for record in lines:
        assert record["request_id"] == request_id
        assert record["tenant_id"] == "t-1"
        assert record["user_id"] == "u-1"


def test_context_is_restored_not_cleared(json_logs: io.StringIO) -> None:
    """嵌套上下文退出后必须还原外层，而不是全清空。

    全清空会让内层任务结束后，外层后续日志失去 request_id，
    表现为"一条请求的日志从中间断成两段、无法关联"。
    """
    log = get_logger("test")
    with request_context() as outer_id:
        with request_context() as inner_id:
            log.info("inner")
        assert outer_id != inner_id
        log.info("after-inner")

    records = [json.loads(raw) for raw in json_logs.getvalue().splitlines() if raw.strip()]
    assert records[-1]["request_id"] == outer_id


def test_bind_context_enriches_existing_context(json_logs: io.StringIO) -> None:
    """鉴权后才拿到 tenant_id 的场景：允许中途补充字段。"""
    log = get_logger("test")
    with request_context():
        bind_context(tenant_id="t-late")
        log.info("after-auth")

    assert _last_json_line(json_logs)["tenant_id"] == "t-late"


def test_context_does_not_leak_between_requests(json_logs: io.StringIO) -> None:
    """上下文绝不能跨请求泄漏——那等于把 A 租户的 id 打进 B 租户的日志。"""
    log = get_logger("test")
    with request_context(tenant_id="tenant-a"):
        log.info("req-a")
    with request_context(tenant_id="tenant-b"):
        log.info("req-b")

    records = [json.loads(raw) for raw in json_logs.getvalue().splitlines() if raw.strip()]
    assert records[-2]["tenant_id"] == "tenant-a"
    assert records[-1]["tenant_id"] == "tenant-b"


def test_request_id_is_unique_and_hex() -> None:
    """request_id 必须无碰撞风险且能安全塞进 HTTP header。"""
    ids = {new_request_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(len(rid) == 32 and rid.isalnum() for rid in ids)


# ----------------------------------------------------- 跨线程 / 跨 Task


def test_context_propagates_into_thread_pool(json_logs: io.StringIO) -> None:
    """run_in_executor 默认不传播 contextvars，必须显式包装。

    解析/OCR 都在线程池跑；不传播的话这些日志全丢 request_id，
    线上排查时下游日志凭空断掉。
    """
    log = get_logger("test")

    def blocking_work() -> None:
        log.info("from-thread")

    async def scenario() -> None:
        with request_context() as request_id:
            call = contextual_call(blocking_work)
            await asyncio.get_running_loop().run_in_executor(None, call)
        return request_id

    expected = asyncio.run(scenario())
    assert _last_json_line(json_logs)["request_id"] == expected


def test_context_propagates_into_asyncio_task(json_logs: io.StringIO) -> None:
    """asyncio.create_task 会自动继承上下文，这是唯一"免费"的场景。"""
    log = get_logger("test")

    async def child() -> None:
        log.info("from-task")

    async def scenario() -> str:
        with request_context() as request_id:
            await asyncio.create_task(child())
        return request_id

    expected = asyncio.run(scenario())
    assert _last_json_line(json_logs)["request_id"] == expected


# ------------------------------------------------------------ 输出格式


def test_json_output_is_single_line_and_parseable(json_logs: io.StringIO) -> None:
    """每行必须是独立可解析的 JSON，否则日志采集器会丢数据。"""
    log = get_logger("test")
    with request_context():
        log.info("multi\nline\nevent", payload={"nested": {"deep": [1, 2, 3]}})

    payload = _last_json_line(json_logs)
    assert payload["event"] == "multi\nline\nevent"  # 换行被转义在同一个 JSON 里
    assert payload["payload"]["nested"]["deep"] == [1, 2, 3]  # type: ignore[index]


def test_third_party_logging_is_bridged(json_logs: io.StringIO) -> None:
    """uvicorn/sqlalchemy 的 stdlib 日志必须也被渲染成同样的 JSON。"""
    logging.getLogger("uvicorn.error").warning("third party message")

    record = _last_json_line(json_logs)
    assert record["event"] == "third party message"
    assert record["level"] == "warning"


def test_repeated_setup_does_not_duplicate_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """重复调用 setup_logging 不能产生重复日志行。

    同样需要隔离环境变量（见 json_logs fixture 的说明）。
    """
    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)

    stream = io.StringIO()
    settings = Settings(
        env="prod",
        log_format="json",
        database={"password": "a-real-strong-secret"},
        identity={"jwt_secret": "a-production-grade-secret-value-32+", "jwt_algorithm": "RS256"},
        llm={"api_key": "sk-a-real-production-key-value"},
        # Embedding 现在是远程 API：生产守卫要求有 key
        embedding={"api_key": "a-real-embedding-key"},
    )
    setup_logging(settings, stream=stream)
    setup_logging(settings, stream=stream)

    get_logger("test").info("once")

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert sum("once" in line for line in lines) == 1
    clear_context()
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()
