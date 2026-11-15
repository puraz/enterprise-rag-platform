"""A14 可观测性测试。

## 三个重点

1. **脱敏必须在出口做**（安全要求，不是可选）
   `docs/PROMPT.md` 明确要求 "Langfuse 打点与节点 span（**含脱敏**）"。
   追踪系统是旁路：数据发到另一个服务、保留更久、访问控制更松。

2. **指标 label 基数必须受控**
   给 document_id/query 打 label 会让时间序列数量随数据量爆炸，
   Prometheus 内存被打爆。这是最常见的生产事故之一。

3. **追踪失败不能影响业务**
   追踪是旁路能力。它挂掉不该让问答失败——但也不能静默，
   否则"追踪没数据"无从排查。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rag_core.metrics import (
    LATENCY_BUCKETS,
    build_metrics,
    record_cache_op,
    record_citation,
    record_llm_call,
    record_ratelimit_denied,
    record_retrieval,
    record_task,
    render_metrics,
    update_breaker_state,
)
from rag_core.tracing import (
    LocalTracer,
    build_tracer,
    redact_attributes,
    text_fingerprint,
    trace_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ==================================================== 脱敏（安全关键）


def test_redacts_sensitive_keys_entirely() -> None:
    """敏感键的值整值替换，不保留内容。

    为什么按**键名**而不是只靠内容规则：内容级 PII 规则只能识别
    格式化数据（手机号/身份证）。用户名、内部项目代号、客户简称
    没有固定格式，规则抓不到，但键名能表达"这是用户输入"。

    断言用"原文不出现"而不是硬编码 `<redacted:N chars>` 的 N：
    后者只要改一下测试数据就得跟着改，且改错了（数错字数）
    会让人误以为脱敏逻辑有 bug。
    """
    payload = {
        "prompt": "请告诉我张三的薪资",
        "query": "张三今年多少岁",
        "answer": "他今年 35 岁",
        "api_key": "sk-1234567890abcdef",
    }
    attrs = redact_attributes({**payload, "model": "deepseek-chat"})

    for key, original in payload.items():
        assert original not in str(attrs[key]), f"{key} 的内容泄漏了"
        assert attrs[key].startswith("<redacted:"), f"{key} 未被脱敏"
        # 保留长度信息（便于判断"prompt 有多大"），但只保留长度
        assert str(len(original)) in attrs[key]

    # 非敏感键保留
    assert attrs["model"] == "deepseek-chat"


def test_redacts_pii_in_non_sensitive_values() -> None:
    """非敏感键的字符串值也要过 PII 规则。

    `path` / `error_message` 这类字段不是敏感键，但完全可能含手机号
    （`log.info(f"用户 {phone} 的文件损坏")` 是很自然的写法）。
    """
    attrs = redact_attributes(
        {
            "error_message": "用户 13812345678 的文档解析失败",
            "model": "deepseek-chat",
        }
    )

    assert "13812345678" not in str(attrs["error_message"])
    assert "138" in str(attrs["error_message"])  # 保留前 3 位便于核对


def test_redacts_nested_structures() -> None:
    """嵌套 dict / list 也要脱敏（span 属性可能是结构化数据）。"""
    attrs = redact_attributes(
        {
            "metadata": {
                "prompt": "秘密内容",
                "nested": [{"content": "更深层的内容"}, "正常文本"],
                "count": 42,
            }
        }
    )

    assert attrs["metadata"]["prompt"] == "<redacted:4 chars>"
    assert attrs["metadata"]["nested"][0]["content"] == "<redacted:6 chars>"
    assert attrs["metadata"]["count"] == 42


def test_keeps_non_sensitive_scalars_intact() -> None:
    """数字/布尔/None 原样保留（脱敏不能破坏可聚合性）。"""
    attrs = redact_attributes({"limit": 20, "is_fallback": True, "score": None})
    assert attrs == {"limit": 20, "is_fallback": True, "score": None}


def test_text_fingerprint_is_stable_and_irreversible() -> None:
    """指纹用于"是否同一份内容"，必须稳定、不可逆、不泄漏长度。"""
    a = text_fingerprint("同一段文本")
    b = text_fingerprint("同一段文本")
    c = text_fingerprint("另一段文本")

    assert a == b
    assert a != c
    assert len(a) == 12
    assert "同一段文本" not in a


def test_sensitive_key_matching_is_case_insensitive() -> None:
    """`Prompt` / `API_KEY` 也要被识别（配置常写成大写下划线）。"""
    attrs = redact_attributes({"Prompt": "内容", "API_KEY": "sk-xxx"})
    assert attrs["Prompt"].startswith("<redacted:")
    assert attrs["API_KEY"].startswith("<redacted:")


# ==================================================== span 结构


async def test_local_tracer_records_parent_child() -> None:
    """span 必须有父子关系（这是归因的前提）。"""
    tracer = LocalTracer()

    async with tracer.aspan("chat.request", question="测试"):
        async with tracer.aspan("retrieval.vector", limit=10):
            await asyncio.sleep(0.001)
        async with tracer.aspan("llm.generate", model="deepseek-chat"):
            await asyncio.sleep(0.001)

    spans = tracer.spans
    assert len(spans) == 3

    root = next(s for s in spans if s.name == "chat.request")
    children = [s for s in spans if s.parent_id == root.span_id]
    assert {s.name for s in children} == {"retrieval.vector", "llm.generate"}
    assert root.parent_id is None
    # 同一 trace
    assert all(s.trace_id == root.trace_id for s in spans)


async def test_span_attributes_are_redacted() -> None:
    """写入 span 的属性在存储前就已脱敏（不在读取时才脱）。"""
    tracer = LocalTracer()
    async with tracer.aspan("test", prompt="秘密问题", model="m1"):
        pass

    span = tracer.spans[0]
    assert span.attributes["prompt"].startswith("<redacted:")
    assert span.attributes["model"] == "m1"


async def test_span_records_duration() -> None:
    tracer = LocalTracer()
    async with tracer.aspan("slow"):
        await asyncio.sleep(0.02)

    assert tracer.spans[0].duration_ms >= 15


async def test_span_ends_on_exception() -> None:
    """异常路径也要结束 span（否则会漏记失败节点的耗时）。"""
    tracer = LocalTracer()
    with pytest.raises(RuntimeError):
        async with tracer.aspan("failing"):
            raise RuntimeError("boom")

    assert len(tracer.spans) == 1
    assert tracer.spans[0].duration_ms >= 0


async def test_tracer_respects_max_spans() -> None:
    """上限保护：追踪不该无限吃内存（长跑的 worker 尤其要注意）。"""
    tracer = LocalTracer(max_spans=5)
    for _ in range(20):
        async with tracer.aspan("op"):
            pass

    assert len(tracer.spans) == 5


def test_trace_summary_aggregates_by_name() -> None:
    """同名 span 按名聚合（多查询会有 3 个 retrieval.vector）。"""
    from rag_core.tracing import SpanData

    spans = [
        SpanData(name="root", trace_id="t", span_id="r", start_ns=0, end_ns=10_000_000),
        SpanData(
            name="retrieval.vector",
            trace_id="t",
            span_id="a",
            parent_id="r",
            start_ns=0,
            end_ns=3_000_000,
        ),
        SpanData(
            name="retrieval.vector",
            trace_id="t",
            span_id="b",
            parent_id="r",
            start_ns=0,
            end_ns=5_000_000,
        ),
    ]

    summary = trace_summary(spans)
    assert summary["span_count"] == 3
    assert summary["total_ms"] == pytest.approx(10.0)

    vector_node = next(n for n in summary["nodes"] if n["name"] == "retrieval.vector")
    assert vector_node["count"] == 2
    assert vector_node["total_ms"] == pytest.approx(8.0)
    assert vector_node["avg_ms"] == pytest.approx(4.0)
    assert vector_node["max_ms"] == pytest.approx(5.0)


def test_build_tracer_without_credentials_falls_back_to_local() -> None:
    """启用追踪但没凭证 → LocalTracer（配置错误要说出来，不能崩）。"""
    tracer = build_tracer(enabled=True)  # 无 public_key/secret
    assert isinstance(tracer, LocalTracer)


def test_build_tracer_disabled_returns_local() -> None:
    tracer = build_tracer(enabled=False)
    assert isinstance(tracer, LocalTracer)


# ==================================================== 指标


def test_metrics_render_in_prometheus_format() -> None:
    """渲染结果必须能被 Prometheus 解析（`# HELP` / `# TYPE` 头齐全）。"""
    metrics = build_metrics()
    record_llm_call(
        metrics,
        task="chat",
        provider="deepseek",
        model="deepseek-chat",
        status="success",
        duration_s=1.5,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0001,
    )

    body = render_metrics(metrics).decode("utf-8")
    assert "# HELP rag_llm_calls_total" in body
    assert "# TYPE rag_llm_calls_total counter" in body
    assert "# TYPE rag_llm_duration_seconds histogram" in body


def test_metrics_include_all_four_llm_signals() -> None:
    """一次 LLM 调用必须同时更新计数/延迟/token/成本四个指标。

    分散写必然有人漏——漏成本指标会让"成本看板"与"对账报告"
    数字不一致，而这类不一致会让人怀疑两个数字都不可信。
    """
    metrics = build_metrics()
    record_llm_call(
        metrics,
        task="chat",
        provider="deepseek",
        model="deepseek-chat",
        status="success",
        duration_s=1.0,
        input_tokens=1000,
        output_tokens=500,
        cached_tokens=200,
        cost_usd=0.0007,
    )

    body = render_metrics(metrics).decode("utf-8")
    assert 'rag_llm_calls_total{is_fallback="false"' in body
    assert "rag_llm_duration_seconds_count" in body
    assert 'rag_llm_tokens_count{kind="input"' in body
    assert 'rag_llm_tokens_count{kind="cached"' in body
    assert "rag_llm_cost_usd_count" in body


def test_fallback_calls_are_labeled() -> None:
    """降级调用必须可区分（否则算不出降级比例，告警无从写）。"""
    metrics = build_metrics()
    record_llm_call(
        metrics,
        task="chat",
        provider="backup",
        model="backup-model",
        status="success",
        duration_s=0.5,
        is_fallback=True,
    )

    body = render_metrics(metrics).decode("utf-8")
    assert 'is_fallback="true"' in body


def test_breaker_state_uses_numeric_values() -> None:
    """熔断器状态用数值（0/1/2），不是字符串。

    Prometheus 的 Gauge 只支持数值。用 label 表达状态会产生
    "一条序列 + 变化的 label"（实际是三条序列），查询时要 filter，
    不如直接用数值 + 固定阈值告警（`> 1` 表示熔断）。
    """
    metrics = build_metrics()
    update_breaker_state(metrics, provider="deepseek", state="open")

    body = render_metrics(metrics).decode("utf-8")
    assert 'rag_llm_breaker_state{provider="deepseek"} 2.0' in body


def test_unknown_breaker_state_is_ignored() -> None:
    """未知状态不写指标（避免产生无意义的 0，被误读为 closed）。"""
    metrics = build_metrics()
    update_breaker_state(metrics, provider="deepseek", state="exploded")

    body = render_metrics(metrics).decode("utf-8")
    assert 'rag_llm_breaker_state{provider="deepseek"}' not in body


def test_no_high_cardinality_labels() -> None:
    """**核心约束**：所有指标的 label 名里不得有 ID / 用户输入类字段。

    这条测试的价值：新增指标时如果手滑加了 `document_id`，
    这个测试会立刻失败——而不是等到生产上 Prometheus 内存被打爆。
    """
    metrics = build_metrics()
    forbidden = {
        "document_id",
        "doc_id",
        "user_id",
        "tenant_id",
        "chunk_id",
        "query",
        "question",
        "prompt",
        "trace_id",
        "request_id",
        "path",
        "url",
    }

    for metric in metrics.registry.collect():
        for sample in metric.samples:
            label_names = set(sample.labels)
            offenders = label_names & forbidden
            assert not offenders, (
                f"指标 {sample.name} 含高基数 label {offenders}：会让时间序列数量随数据量爆炸"
            )


def test_route_label_uses_template_not_actual_path() -> None:
    """HTTP 指标的 route 必须是路由模板（不是实际路径）。

    实际路径含 UUID，十万文档 = 十万条序列。
    """
    from fastapi.testclient import TestClient

    from rag_api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        client.get("/health/live")
        client.get("/api/v1/does-not-exist")

        body = client.get("/health/metrics").text
        http_lines = [
            line for line in body.splitlines() if line.startswith("rag_http_requests_total")
        ]

    assert any('route="/health/live"' in line for line in http_lines)
    # 未匹配的路径不出现（否则扫描器会让基数爆炸）
    assert any('route="__unmatched__"' in line for line in http_lines)
    assert not any("/api/v1/does-not-exist" in line for line in http_lines)


def test_latency_buckets_cover_realistic_range() -> None:
    """延迟桶必须覆盖 5ms~30s。

    桶太少会让 P95/P99 只能落在粗粒度边界上——"优化了 200ms"
    在指标上看不出来。
    """
    assert min(LATENCY_BUCKETS) <= 0.005
    assert max(LATENCY_BUCKETS) >= 30.0
    # 缓存命中/向量检索在 5~50ms，需要细桶
    assert any(0.01 <= b <= 0.05 for b in LATENCY_BUCKETS)


def test_record_helpers_update_multiple_metrics() -> None:
    """便捷记录函数一次更新多个指标。"""
    metrics = build_metrics()
    record_retrieval(metrics, strategy="vector", duration_s=0.05, result_count=10)
    record_cache_op(metrics, namespace="qa", result="hit")
    record_task(metrics, task_type="parse_document", outcome="succeeded", duration_s=1.2)
    record_citation(metrics, located=True)
    record_ratelimit_denied(metrics, policy="tenant_chat")

    body = render_metrics(metrics).decode("utf-8")
    assert 'rag_retrieval_total{strategy="vector"}' in body
    assert 'rag_cache_operations_total{namespace="qa",result="hit"}' in body
    assert 'rag_tasks_processed_total{outcome="succeeded"' in body
    assert 'rag_citations_total{located="located"}' in body
    assert 'rag_ratelimit_denied_total{policy="tenant_chat"}' in body


def test_metrics_registry_is_isolated_per_instance() -> None:
    """每个 Metrics 实例有独立 registry（测试可以拿干净的）。

    用全局默认 registry 的话，测试间会累积注册，
    重复创建会抛 `Duplicated timeseries`。
    """
    first = build_metrics()
    second = build_metrics()

    record_llm_call(
        first,
        task="chat",
        provider="p",
        model="m",
        status="success",
        duration_s=0.1,
    )

    first_body = render_metrics(first).decode("utf-8")
    second_body = render_metrics(second).decode("utf-8")

    assert "rag_llm_calls_total{" in first_body
    # 第二个实例不该有数据
    assert "rag_llm_calls_total{task=" not in second_body


# ==================================================== 部署产物


def test_alert_rules_are_valid_yaml() -> None:
    """告警规则必须是合法 YAML 且每条都有实际内容。

    YAML 一个缩进错就整份规则加载失败——Prometheus 只会打一行日志，
    很容易被忽略（表现为"告警永远不触发"）。
    """
    yaml = pytest.importorskip("yaml")
    path = REPO_ROOT / "deploy/observability/prometheus/alerts.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert "groups" in data
    alert_count = 0
    for group in data["groups"]:
        assert "name" in group
        for rule in group["rules"]:
            alert_count += 1
            assert rule["alert"], "告警必须有名字"
            assert rule["expr"], f"{rule['alert']} 缺 expr"
            assert rule.get("labels", {}).get("severity") in {"critical", "warning", "info"}
            # 每条告警都要有 summary（通知里显示的内容）
            assert rule.get("annotations", {}).get("summary")

    assert alert_count >= 8, "告警规则太少（应覆盖可用性/延迟/LLM/成本/缓存）"


def test_dashboard_json_is_valid() -> None:
    """看板 JSON 必须可解析且面板结构完整。"""
    path = REPO_ROOT / "deploy/observability/grafana/dashboard.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["title"]
    assert data["panels"], "看板不能没有面板"
    assert data["refresh"] == "30s"

    for panel in data["panels"]:
        assert panel["title"], "每个面板都要有标题"
        assert panel["targets"], f"{panel['title']} 缺查询"
        assert panel["targets"][0]["expr"], f"{panel['title']} 的查询为空"
        # 面板必须落在栅格内（Grafana 用 24 列）
        pos = panel["gridPos"]
        assert pos["x"] + pos["w"] <= 24, f"{panel['title']} 超出 24 列"
