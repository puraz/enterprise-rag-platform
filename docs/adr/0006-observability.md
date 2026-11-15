# A14 可观测性：脱敏在出口、基数受控、告警有依据

> 日期：2026-11-15　环境：Prometheus 客户端 + 可选 Langfuse
> 复现：`uv run python tests/... `（见 `tests/test_observability.py`，26 项）

## 1. 三条设计原则

### 原则一：脱敏必须在出口做，且默认拒绝

`docs/PROMPT.md` 要求 "Langfuse 打点与节点 span（**含脱敏**）"。

追踪系统的危害在于它是**旁路**：
- 数据发到另一个服务（可能是 SaaS）；
- 那里通常保留更久、访问控制更松；
- 很容易被当成"调试数据"而豁免合规审查。

实现（`rag_core.tracing.redact_attributes`）：

| 层级 | 规则 |
|---|---|
| 键名命中敏感词（`prompt`/`query`/`answer`/`api_key`…） | 整值替换为 `<redacted:N chars>` |
| 其它字符串值 | 过 PII 规则（手机号/身份证/邮箱/银行卡…） |
| 数字/布尔/None | 原样保留（脱敏不能破坏可聚合性） |

**为什么按键名脱敏而不只靠内容规则**：内容级 PII 规则只能识别格式化数据。
用户名、内部项目代号、客户简称没有固定格式，规则抓不到，
但键名能表达"这是用户输入"。

保留长度的理由：`<redacted:1234 chars>` 能判断"prompt 有多大"，
对排查"上下文爆炸导致成本飙升"有用，且不泄漏内容。

另提供 `text_fingerprint()`（SHA-256 前 12 位）：用于判断
"不同 span 是否处理同一份内容"，不可逆，比记录正文安全得多。

### 原则二：label 基数必须受控

给 `document_id` / `query` / `user_id` 打 label 会让时间序列数量
随数据量线性增长——十万文档 = 十万条 `http_requests_total` 序列，
Prometheus 内存被打爆。这是最常见的生产事故之一。

实现：
- 所有指标 label 都是低基数（状态、模型名、任务类型、错误码）；
- HTTP 的 `route` 用**路由模板**（`/api/v1/documents/{doc_id}`）
  而不是实际路径；
- **404 统一标成 `__unmatched__`**，不用实际路径——扫描器会用
  随机路径探测（`/.env`、`/wp-admin`），每个都成一条新序列。

实测验证（`test_route_label_uses_template_not_actual_path`）：

```
rag_http_requests_total{method="GET",route="/health/live",status="200"} 2.0
rag_http_requests_total{method="GET",route="__unmatched__",status="404"} 2.0
```

扫描器路径 `/.env` 没有出现在任何 label 里。

还有一条防御性测试 `test_no_high_cardinality_labels`：遍历所有指标的
所有 label，断言不含 `document_id`/`user_id`/`query`/`path` 等 13 个禁词。
新增指标时手滑加了高基数 label，这个测试立刻失败——而不是等生产事故。

### 原则三：每条告警都要能回答"我该做什么"

告警的失败模式不是"不够灵敏"，而是"太多噪音让人开始忽略"。

10 条规则，每条都注明：
- **阈值依据**（"P95 > 2s：非 LLM 接口应在 500ms 内"）；
- **看到后该查什么**（`description` 里的排查步骤）；
- **severity 分级**（critical 立即处理 / warning 当天 / info 观察）。

| 组 | 规则 |
|---|---|
| 可用性 | `APIDown`、`HighErrorRate` |
| 延迟 | `HighAPILatency`、`SlowRetrieval` |
| LLM | `LLMBreakerOpen`、`LLMErrorRateHigh`、`LLMDegradedToFallback` |
| 成本 | `HighTokenUsage`、`CostSpike` |
| 缓存 | `LowCacheHitRate` |

### 一处被自己写坏又改掉的地方

`LLMDegradedToFallback` 的第一版写成了：

```yaml
expr: |
  (sum(rate(...)) and on() group_left() (sum(rate(...)) > 0)) * 0
  + (sum(rate(rag_llm_cost_usd_count[5m]))) * 0
```

这是无意义表达式（乘 0 后恒为 0，永远不会触发）。根因是我当时
**没有 `is_fallback` 这个 label**，于是想"绕过去先留个占位"。

正确做法是**先补 label 再写规则**：
`rag_llm_calls_total` 新增 `is_fallback`（布尔，2 个基数），
然后用直白的比例表达式：

```yaml
expr: |
  sum(rate(rag_llm_calls_total{is_fallback="true"}[5m]))
  / sum(rate(rag_llm_calls_total[5m])) > 0.2
```

**不要写"占位用的假表达式"**——它看起来像一条规则，实际永远不告警，
比缺失更危险。

## 2. 12 个指标 + 22 个面板

指标按四类（Counter / Histogram / Gauge / 计数聚合）：

| 域 | 指标 |
|---|---|
| HTTP | `rag_http_requests_total`、`_duration_seconds`、`_errors_total` |
| 检索 | `rag_retrieval_total`、`_duration_seconds`、`_results_count` |
| LLM | `rag_llm_calls_total`、`_duration_seconds`、`_tokens`、`_cost_usd`、`_breaker_state` |
| 引用 | `rag_citations_total` |
| 限流 | `rag_ratelimit_denied_total` |
| Worker | `rag_tasks_processed_total`、`rag_tasks_inflight`、`_duration_seconds` |
| 缓存 | `rag_cache_operations_total`、`_hit_ratio` |

看板（`deploy/observability/grafana/dashboard.json`）六层结构：

1. 核心健康度（6 个 stat：速率、错误率、P95、熔断数、缓存命中、降级占比）
2. HTTP 层（按路由的速率与延迟、错误码分布）
3. 检索层（按策略的延迟、结果数分布）
4. LLM 层（按任务的速率与延迟、失败率、token 用量）
5. 成本层（单次成本 P95、成本速率）
6. Worker 与限流（任务速率、耗时、被限流数、引用定位成功率）

面板的 `description` 写了"异常时该看什么"——例如检索延迟面板写着
"慢于 500ms 通常说明 HNSW 索引没被用上（见 ADR 0002）"。

## 3. 为什么 worker 指标暂不抓取

Worker 没有 HTTP 端点。两种方案各有代价：

- **多进程模式**（`PROMETHEUS_MULTIPROC_DIR`）：各进程写共享目录，
  需要 sidecar 暴露。Gauge 的语义在多进程下变模糊
  （"单进程在跑几个"还是"全部进程合计"？）。
- **Pushgateway**：适合批处理，但"进程消失后指标残留"会让
  `tasks_inflight` 之类的 Gauge 永久显示假值。

代码已支持多进程模式（`build_metrics()` 检测环境变量），
但抓取配置里注释掉了 worker job——**留到 A15 压测时按实际部署形态决定**，
而不是现在猜一个。

## 4. 遗留

1. **Langfuse 未实测**：`LangfuseTracer` 已实现（失败自动降级 LocalTracer），
   但本机无 Langfuse 服务端，只验证了降级路径。真实打点需起 Langfuse。
2. **追踪未接入业务链路**：`LocalTracer` 可用于测试，但检索图与
   LLM 调用还没有真正 `span()` 包裹。接入点明确（A11 的
   `RetrievalGraph.run` 与 A5 的 `ModelGateway.complete`），
   但需要一次专门的接线改动。
3. **告警未接入 Alertmanager**：规则写好了，但路由/分组/静默策略
   需要实际部署环境决定。
4. **成本指标与对账可能不一致**：`rag_llm_cost_usd` 是 Histogram（观测分布），
   `model_calls.cost_usd` 是精确求和（对账用）。两者口径不同，
   **不要用指标算账**——账目以数据库为准。
