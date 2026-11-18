"""追踪层单测：脱敏、本地 span 树、Langfuse SDK 版本契约。

## 为什么要有"假 SDK"这一组测试

`LangfuseTracer` 里所有 SDK 调用都被 try/except 包着（追踪不能影响业务）。
没有测试锁住接口名的话，SDK 升级把 `start_as_current_span` 换成
`start_as_current_observation` 这类变化**不会让任何测试失败**——
真实环境里表现是"Langfuse 一条数据都没有"，业务侧风平浪静。

这不是假设：本项目从 v3 风格接口迁到 langfuse 5.x 时，
`start_as_current_span` 已经是 `None`，每次开 span 都抛 AttributeError
并被吞掉，全部降级成本地内存记录。外部无法发现，只能靠这组测试。
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from typing import Any, ClassVar

import httpx
import pytest

from rag_core.settings import Settings
from rag_core.settings.tracing import TracingSettings
from rag_core.tracing import (
    LangfuseTracer,
    LocalTracer,
    SpanData,
    build_tracer,
    redact_attributes,
    text_fingerprint,
    trace_summary,
)

# ---------------------------------------------------------------- 脱敏


def test_redact_sensitive_keys_keep_length_only() -> None:
    """敏感键只保留长度：能判断"prompt 有多大"，但不泄漏内容。"""
    redacted = redact_attributes(
        {
            "prompt": "我的身份证号是 11010519491231002X",
            "answer": "答案正文",
            "api_key": "sk-live-should-never-appear",
        }
    )

    assert redacted["prompt"].startswith("<redacted:")
    assert redacted["prompt"].endswith(" chars>")
    # 长度标记里的数字必须是**原文字符数**（便于判断"这份 prompt 有多大"）
    assert f"<redacted:{len('我的身份证号是 11010519491231002X')} chars>" == redacted["prompt"]
    assert redacted["answer"] == "<redacted:4 chars>"
    # 断言原文不出现（不硬编码具体字面量，避免数错字数时误判）
    assert "11010519491231002X" not in str(redacted)
    assert "sk-live" not in str(redacted)


def test_redact_non_sensitive_keys_still_applies_pii_rules() -> None:
    """键名不敏感 ≠ 值不敏感：路径里的手机号也要打码。"""
    redacted = redact_attributes({"path": "/用户/13800138000/报告.pdf"})
    assert "13800138000" not in redacted["path"]
    assert "138" in redacted["path"]  # 保留前 3 位便于核对


def test_redact_nested_structures() -> None:
    redacted = redact_attributes({"meta": {"prompt": "x" * 10, "n": 3}})
    assert redacted["meta"]["prompt"] == "<redacted:10 chars>"
    assert redacted["meta"]["n"] == 3


def test_text_fingerprint_is_stable_and_short() -> None:
    assert text_fingerprint("同一段内容") == text_fingerprint("同一段内容")
    assert text_fingerprint("a") != text_fingerprint("b")
    assert len(text_fingerprint("a")) == 12


# ---------------------------------------------------------------- LocalTracer


def test_local_tracer_builds_span_tree() -> None:
    tracer = LocalTracer()

    with tracer.span("root") as root, tracer.span("child") as child:
        pass

    assert len(tracer.spans) == 2
    assert isinstance(root, SpanData)
    assert child.trace_id == root.trace_id
    assert child.parent_id == root.span_id
    assert root.parent_id is None
    assert root.duration_ms >= 0


@pytest.mark.asyncio
async def test_local_tracer_async_spans_and_annotate() -> None:
    tracer = LocalTracer()

    async with tracer.atrace("chat.turn", user_id="u1", session_id="s1") as root:
        async with tracer.aspan("retrieval.vector", limit=20) as span:
            tracer.annotate(span, hits=3)
        async with tracer.ageneration("llm.stream", provider="deepseek") as gen:
            tracer.record_usage(gen, input_tokens=10, output_tokens=5, cost_usd=0.0001)

    spans = {span.name: span for span in tracer.spans}
    assert spans["retrieval.vector"].attributes["hits"] == 3
    assert spans["llm.stream"].kind == "generation"
    assert spans["llm.stream"].attributes["input_tokens"] == 10
    assert spans["llm.stream"].attributes["cost_usd"] == 0.0001
    assert all(span.trace_id == root.trace_id for span in tracer.spans)


@pytest.mark.asyncio
async def test_local_tracer_isolates_concurrent_tasks() -> None:
    """并发任务不能互相成为父节点（初版用实例属性踩过的坑）。"""
    import asyncio

    tracer = LocalTracer()

    async def one_turn(name: str) -> str:
        async with tracer.atrace(name) as root:
            await asyncio.sleep(0)
            async with tracer.aspan(f"{name}.child") as child:
                await asyncio.sleep(0)
        return child.trace_id if child.trace_id == root.trace_id else "mismatch"

    results = await asyncio.gather(*(one_turn(f"turn{i}") for i in range(5)))
    assert len(set(results)) == 5  # 每个任务各自一条 trace，且自身父子一致

    roots = [span for span in tracer.spans if span.kind == "trace"]
    assert len({root.trace_id for root in roots}) == 5


def test_trace_summary_aggregates_by_name() -> None:
    tracer = LocalTracer()
    with tracer.span("retrieval.vector"):
        pass
    with tracer.span("retrieval.vector"):
        pass
    with tracer.span("retrieval.keyword"):
        pass

    summary = trace_summary(tracer.spans)
    nodes = {node["name"]: node for node in summary["nodes"]}
    assert nodes["retrieval.vector"]["count"] == 2
    assert nodes["retrieval.keyword"]["count"] == 1
    assert summary["span_count"] == 3


# ---------------------------------------------------------------- 配置


def test_tracing_settings_requires_protocol_in_host() -> None:
    with pytest.raises(ValueError, match="必须带协议"):
        TracingSettings(host="localhost:3030")


def test_tracing_settings_defaults_to_3030_not_grafana_3000() -> None:
    """默认端口不能是 3000：那是 Grafana，撞了会导致追踪静默失效。"""
    assert TracingSettings().host == "http://localhost:3030"


def test_tracing_settings_configured_requires_both_keys() -> None:
    assert not TracingSettings(enabled=True).configured
    assert not TracingSettings(enabled=True, public_key="pk").configured
    assert not TracingSettings(public_key="pk", secret_key="sk").configured
    assert TracingSettings(enabled=True, public_key="pk", secret_key="sk").configured


def test_tracing_settings_summary_never_leaks_keys() -> None:
    summary = TracingSettings(
        enabled=True, public_key="pk-lf-secret", secret_key="sk-lf-secret"
    ).redacted_summary
    assert "pk-lf-secret" not in str(summary)
    assert "sk-lf-secret" not in str(summary)
    assert summary["has_secret_key"] is True


def test_settings_reject_unknown_tracing_field() -> None:
    """`extra=forbid`：拼错配置项要立刻报错，而不是静默忽略。"""
    with pytest.raises(ValueError):
        Settings(tracing={"enabled": True, "publik_key": "typo"})


# ---------------------------------------------------------------- 装配


def test_build_tracer_without_settings_returns_local() -> None:
    assert isinstance(build_tracer(None), LocalTracer)


def test_build_tracer_disabled_returns_local() -> None:
    tracer = build_tracer(TracingSettings(enabled=False))
    assert isinstance(tracer, LocalTracer)


def test_build_tracer_enabled_without_credentials_returns_local() -> None:
    tracer = build_tracer(TracingSettings(enabled=True, host="http://localhost:3030"))
    assert isinstance(tracer, LocalTracer)


# ---------------------------------------------------------------- 假 SDK


class _FakeObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeContextManager:
    """**只实现同步协议**，刻意不实现 async 版本。

    langfuse 5.2 的 `start_as_current_observation` 返回的
    `_AgnosticContextManager` 实际只支持 `with`；初版在 `aspan` 里写
    `async with`，导致 worker 每个任务都抛 TypeError 进死信。
    假实现只给同步协议，才能锁住这个约束。
    """

    def __init__(self, observation: _FakeObservation) -> None:
        self.observation = observation

    def __enter__(self) -> _FakeObservation:
        return self.observation

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeLangfuse:
    instances: ClassVar[list[_FakeLangfuse]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.opened: list[tuple[str, str, Any]] = []
        self.models: list[str] = []
        self.observations: list[_FakeObservation] = []
        self.scores: list[dict[str, Any]] = []
        self.shutdown_called = False
        self.fail_open = False
        _FakeLangfuse.instances.append(self)

    def start_as_current_observation(
        self, *, name: str, as_type: str = "span", metadata: Any = None, model: str | None = None
    ) -> _FakeContextManager:
        if self.fail_open:
            raise RuntimeError("sdk boom")
        self.opened.append((name, as_type, metadata))
        if model is not None:
            self.models.append(model)
        observation = _FakeObservation()
        self.observations.append(observation)
        return _FakeContextManager(observation)

    def get_current_trace_id(self) -> str:
        return "trace-from-sdk"

    def auth_check(self) -> bool:
        return True

    def create_score(self, **kwargs: Any) -> None:
        # 保留这个假接口，但生产代码**不再调用它**（见 score_trace 的注释：
        # SDK 5.2 的 create_score 与服务端 3.227 不兼容，且只打印不抛错）。
        self.scores.append(kwargs)

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture()
def fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLangfuse]:
    """把 `langfuse` 模块替换成假实现，锁住 SDK 接口名。"""
    module = types.ModuleType("langfuse")
    module.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
    module.propagate_attributes = lambda **kwargs: nullcontext()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)
    _FakeLangfuse.instances.clear()
    return _FakeLangfuse


def _settings(**overrides: Any) -> TracingSettings:
    base: dict[str, Any] = {
        "enabled": True,
        "public_key": "pk-lf-test",
        "secret_key": "sk-lf-test",
        "host": "http://langfuse:3030",
        "environment": "test",
        "release": "1.2.3",
        "sample_rate": 0.5,
    }
    base.update(overrides)
    return TracingSettings(**base)


def test_langfuse_tracer_passes_config_to_sdk(fake_langfuse: type[_FakeLangfuse]) -> None:
    tracer = build_tracer(_settings())
    assert isinstance(tracer, LangfuseTracer)

    client = fake_langfuse.instances[-1]
    assert client.kwargs["host"] == "http://langfuse:3030"
    assert client.kwargs["environment"] == "test"
    assert client.kwargs["release"] == "1.2.3"
    assert client.kwargs["sample_rate"] == 0.5
    assert client.kwargs["tracing_enabled"] is True


@pytest.mark.asyncio
async def test_langfuse_tracer_uses_v5_observation_api(
    fake_langfuse: type[_FakeLangfuse],
) -> None:
    """span/generation 必须走 `start_as_current_observation`（v5 的唯一接口）。"""
    tracer = build_tracer(_settings())
    client = fake_langfuse.instances[-1]

    async with tracer.atrace("chat.turn", user_id="u1", session_id="s1") as root:
        async with tracer.aspan("retrieval.vector", hits=2, prompt="机密正文") as span:
            tracer.annotate(span, extra=1)
        async with tracer.ageneration("llm.stream", model="deepseek") as gen:
            tracer.record_usage(gen, input_tokens=7, output_tokens=3, cost_usd=0.5)

    names = [name for name, _kind, _meta in client.opened]
    kinds = {name: kind for name, kind, _meta in client.opened}
    assert names == ["chat.turn", "retrieval.vector", "llm.stream"]
    assert kinds["retrieval.vector"] == "span"
    assert kinds["llm.stream"] == "generation"

    # 脱敏在进入 SDK 之前完成
    vector_meta = dict(client.opened[1][2])
    assert vector_meta["prompt"] == "<redacted:4 chars>"
    assert vector_meta["hits"] == 2

    # generation 的 `model` 必须走专用字段（否则 Langfuse 无法按模型聚合）
    assert "deepseek" in client.models
    assert "model" not in dict(client.opened[2][2] or {})

    assert tracer.current_trace_id() == "trace-from-sdk"
    assert root is not None

    # record_usage / annotate 都通过 update 落到 observation 上
    assert {"usage_details": {"input": 7, "output": 3}, "cost_details": {"total": 0.5}} in (
        client.observations[2].updates
    )
    assert {"metadata": {"extra": 1}} in client.observations[1].updates


@pytest.mark.asyncio
async def test_langfuse_tracer_scores_via_rest_and_shuts_down(
    fake_langfuse: type[_FakeLangfuse], monkeypatch: pytest.MonkeyPatch
) -> None:
    """score 走 `POST /api/public/scores`，不是 SDK 的 `create_score`。

    实测 `langfuse==5.2.0` 对服务端 3.227 调 `create_score` 会 400，
    且只打印不抛错 → 业务侧“成功”、Langfuse 里没数据。这个用例锁住
    REST 路径与请求体形状。
    """
    tracer = build_tracer(_settings())
    client = fake_langfuse.instances[-1]
    captured: dict[str, Any] = {}

    class _FakeHttpClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeHttpClient:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            captured.update(kwargs)
            return httpx.Response(200, json={"id": "score-1"})

    monkeypatch.setattr("rag_core.tracing.httpx.AsyncClient", _FakeHttpClient)

    assert await tracer.averify() is True
    await tracer.score_trace(
        trace_id="trace-1", name="user_feedback", value=0.0, comment="not_relevant"
    )

    assert captured["url"] == "http://langfuse:3030/api/public/scores"
    payload = captured["json"]
    assert payload["traceId"] == "trace-1"
    assert payload["name"] == "user_feedback"
    assert payload["value"] == 0.0
    assert payload["dataType"] == "NUMERIC"
    assert payload["comment"] == "not_relevant"
    # 不再调用 SDK 的 create_score
    assert client.scores == []

    await tracer.aclose()
    # 只 flush 不 shutdown 会丢最后一批 span（后台线程还在跑）
    assert client.shutdown_called is True


@pytest.mark.asyncio
async def test_langfuse_tracer_degrades_when_sdk_open_fails(
    fake_langfuse: type[_FakeLangfuse],
) -> None:
    """SDK 开 span 失败时必须降级到本地记录，且业务不感知。"""
    tracer = build_tracer(_settings())
    client = fake_langfuse.instances[-1]
    client.fail_open = True

    async with tracer.aspan("retrieval.keyword", hits=0) as span:
        assert isinstance(span, SpanData)
        assert span.name == "retrieval.keyword"

    assert tracer.current_trace_id() == "trace-from-sdk"


def test_build_tracer_degrades_when_sdk_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK 未安装时构造降级，不能抛异常。"""
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    tracer = build_tracer(_settings())
    assert isinstance(tracer, LangfuseTracer)
    assert tracer.is_available is False
